#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import re
import glob
import time
import pickle
import numpy as np
import pandas as pd
import librosa
import torch
from argparse import ArgumentParser

from transformers import WhisperProcessor, WhisperConfig
from transformers import WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModel, AutoConfig

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class TranscriptionDatasetInfer(torch.utils.data.Dataset):
    def __init__(self, df, proc_image):
        """
        Dataset for transcription inference. 
        Implements audio loading and preprocessing.

        Parameters
        df: pd.DataFrame
            Data frame
        proc_image
            Whisper processor

        Returns:
        sample : dict
            Dictionary with features representing single example
        """
        self.df = df
        self.proc_image = proc_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # audio
        y, sr = librosa.load(row['file'], sr=24000)
        y = librosa.resample(y, orig_sr=24000, target_sr=16000)
        image = self.proc_image(y, sampling_rate=16000, return_tensors='np').input_features[0]
        sample = {'image': image}
        return sample


class TranscriptionDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, df, proc_image, max_length):  # max_length=23
        """
        Dataset for transcription training. 
        Implements audio and expected text loading and preprocessing.

        Parameters
        df: pd.DataFrame
            Data frame
        proc_image
            Whisper processor
        max_length : int
            Maximum length for Whisper (expected text)

        Returns:
        sample : dict
            Dictionary with features representing single example
        """                
        self.df = df
        self.proc_image = proc_image
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # audio
        y, sr = librosa.load(row['file'], sr=24000)
        y = librosa.resample(y, orig_sr=24000, target_sr=16000)
        image = self.proc_image(y, sampling_rate=16000, return_tensors='np').input_features[0]
        # text
        inputs = self.proc_image.tokenizer(row['expected_text'], 
                                           return_tensors='np', 
                                           truncation=True, 
                                           padding='max_length', 
                                           max_length=self.max_length)
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]        
        # sample
        sample = {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask}
        return sample


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df, gen_voice_map, proc_image, proc_text, max_length):  # max_length=40
        """
        Dataset for classification. Implements audio and text loading and preprocessing.

        Parameters:
        df: pd.DataFrame
            Data frame
        gen_voice_map : dict
            Dictionary with a speech arrays generated from "expected_text"
        proc_image
            Whisper processor
        proc_text
            Deberta tokenizer
        max_length : int
            Maximum length for Deberta (expected text plus transcription)

        Returns:
        sample : dict
            Dictionary with features representing single example
        """
        self.df = df
        self.gen_voice_map = gen_voice_map
        self.proc_image = proc_image
        self.proc_text = proc_text
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load orig audio
        y, sr = librosa.load(row['file'], sr=24000)
        y = librosa.resample(y, orig_sr=24000, target_sr=16000)
        # add generated audio
        y_gen = self.gen_voice_map[row['expected_text']]
        y = np.hstack([y, y_gen])
        image = self.proc_image(y, sampling_rate=16000, return_tensors='np').input_features[0]
        # text
        inputs = self.proc_text(row['expected_text'] + ' ' + row['trans'],
                                return_tensors='np',
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_length)
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        # label        
        if 'score' in row:
            label = np.int64(row['score'])
        else:
            label = -1
        sample = {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
        return sample


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, image_encoder_name, text_encoder_name, hidden_dim=256, tuned_transcriber=None):
        """
        Multimodal classifier. Classifies an example based on audio and text features.

        Parameters:
        image_encoder_name : str
            Whisper architecture name (e.g. "openai/whisper-medium.en")
        text_encoder_name : str
            Deberta architecture name (e.g. "microsoft/deberta-v3-base")
        hidden_dim : int
            Size of the hidden dimension of the classification layer
        tuned_transcriber : str
            Path to a tuned transcriber model file (needed for training)
        """
        super(MultimodalClassifier, self).__init__()
        # Audio
        config = WhisperConfig.from_pretrained(image_encoder_name)
        model = WhisperForConditionalGeneration(config)
        # Init from tuned transcriber
        if tuned_transcriber is not None:
            model.load_state_dict(torch.load(tuned_transcriber, map_location=torch.device('cpu')))
            print('Init from tuned transcriber:', tuned_transcriber)
        self.image_feature_extractor = model.model.encoder
        image_feat_dim = self.image_feature_extractor.config.d_model
        # Text
        config = AutoConfig.from_pretrained(text_encoder_name)
        self.text_feature_extractor = AutoModel.from_config(config)
        text_feat_dim = self.text_feature_extractor.config.hidden_size
        # Fusion
        self.fc_fusion = torch.nn.Linear(image_feat_dim + text_feat_dim, hidden_dim)
        # Cls
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        # Activation
        self.relu = torch.nn.ReLU()

    def forward(self, images, input_ids, attention_mask):
        # Extract audio features
        image_features = self.image_feature_extractor(images).last_hidden_state
        image_features = image_features.mean(dim=1)
        # Extract text features
        text_outputs = self.text_feature_extractor(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        # Fuse
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = self.fc_fusion(combined_features)
        combined_features = self.relu(combined_features)
        # Cls
        output = self.classifier(combined_features)
        return output


def clean(x):
    """
    Clean text line:
        convert to lowercase and strip all outer spaces
        remove any chars except letters and spaces

    Parameters:
    x : str
        Text line

    Returns:
    line : str
        Clean text line
    """
    return re.sub('[^a-zA-Z ]+', '', x.strip().lower())



def keep_n_last_ckpt(wildcard, n=1):
    """
    Sort and remove all checkpoint files except n last.
    If during training we save only checkpoints which improved the score 
    this function retains n best checkpoints.

    Parameters:
    wildcard : str
        Checkpoint path wildcard e.g. 'model-f%d-*' % fold_id
    n : int
        Number of last checkpoints to retain
    """
    assert n > 0, 'Number of files to keep must be > 0'
    files = sorted(glob.glob(wildcard))
    if len(files):
        for file in files[:-n]:
            os.remove(file)


def keep_n_best_ckpt(wildcard, n=1):
    """
    Sort and remove all checkpoint files except n best.

    Parameters:
    wildcard : str
        Checkpoint path wildcard e.g. 'model-f%d-*' % fold_id
    n : int
        Number of best checkpoints to retain
    """
    assert n > 0, 'Number of files to keep must be > 0'
    files = sorted(glob.glob(wildcard), key=lambda x: float(x.split('-')[-3]))
    if len(files):
        for file in files[n:]:
            os.remove(file)


def get_max_length(corpus, tokenizer):
    """
    Computes maximum length (including special tokens) 
    for a given text corpus and tokenizer.

    Note. For WhisperTokenizer there is an addition of 2 to max_length.
    This line was added in early experiments and then not intentionally 
    went into training of the final models. Addition is not needed, 
    but should not affect the result, because it just creates 2 more padding tokens. 
    So I leave it for consistency.    

    Parameters:
    corpus : list of str
        List of text lines
    tokenizer : 
        Tokenizer

    Returns:
    max_length : int
        Maximum length
    """
    lens = []
    for line in corpus:
        lens.append(len(tokenizer(line)['input_ids']))

    max_length = np.max(lens)

    if tokenizer.__class__.__name__ == 'WhisperTokenizer':
        max_length += 2

    return max_length

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


