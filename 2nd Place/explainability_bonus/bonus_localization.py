# Load various libraries
import gc
import re
import os
# In order to run in the competition's runtime environment, some libraries need to be installed.
os.system('python -m ensurepip  > /dev/null') 
os.system('python -m pip install assets/MeloTTS --no-index --find-links=assets/libs  > /dev/null')

import nltk
nltk.data.path.append('./assets')

import copy
import json
import time
import math
import string
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle

import transformers
import torchaudio
import librosa
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from melo.api import TTS

class CFG:
    device='cuda'

# Here is the pooling configuration of the model used in the demo. 
# In fact, different pooling has little effect on the results. The original plan retains these configurations for more diverse model integration.
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, last_hidden_state, attention_mask=None):
        attention_scores = self.attention(last_hidden_state).squeeze(-1)  # (batch_size, seq_len)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), last_hidden_state)  # (batch_size, 1, hidden_size)
        weighted_sum = weighted_sum.squeeze(1)  # (batch_size, hidden_size)
        
        return weighted_sum

class CustomModel(nn.Module):
    def feature(self, audio_tokens, label_audio_tokens):
        audio_features = self.speech_model._mask_input_features(label_audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        label_audio_embedding = self.pool(audio_features)
        label_audio_embedding = self.dropout(label_audio_embedding)
        
        return label_audio_embedding

    def forward(self, audio_tokens, label_audio_tokens, task_ids, grade_ids):
        label_audio_embedding = self.feature(audio_tokens, label_audio_tokens)
        other_features = torch.stack([task_ids, grade_ids], dim=-1)
        label_audio_embedding = torch.cat([label_audio_embedding, other_features], dim=-1)
        output = self.fc1(label_audio_embedding).sigmoid()[:, 0].to(torch.float32)
        return output
    
# First, we load the model and its data processor used in the demo. Here we use the first fold of the first model.
with open('model_configs.pkl', 'rb') as f:
    model_config = pickle.load(f)[0] # the configuration of the first model
CFG.processor = AutoProcessor.from_pretrained(model_config['speech_processor'], chunk_length=model_config['max_speech_len'])
model = torch.load(model_config['paths'][0])['model'] # the first fold
model.to(CFG.device).eval()
for k, v in model.named_parameters():
    dtype = v.dtype
    break

CFG.task2id = {
    'deletion': 0, 
    'nonword_repetition': 1, 
    'blending': 2, 
    'sentence_repetition': 3
}
CFG.grade2id = {
    'KG': 0, 
    '1': 1, 
    '2': 2, 
    '3': 3
}
def processor_func(audio, tts_audios, task, grade):
    input_features = []
    for audio2 in tts_audios:
        audio = np.concatenate([audio2, audio], axis=-1)
        audio = torch.tensor(audio)
        audio_tokens = CFG.processor.feature_extractor(audio, sampling_rate=16000)
        audio_tokens = torch.tensor(audio_tokens['input_features'][0])
        input_features.append(audio_tokens)
    audio_tokens = {'input_features': torch.stack(input_features, dim=0)}

    task_id = CFG.task2id[task]
    grade_id = CFG.grade2id[grade]
    task_id = torch.ones([len(tts_audios)]) * task_id / len(CFG.task2id)
    grade_id = torch.ones([len(tts_audios)]) * grade_id / len(CFG.grade2id)
    return audio_tokens, task_id, grade_id

# Load TTS model
tts_model = TTS(language='EN_NEWEST', device=CFG.device, use_hf=False,
            config_path='assets/tts_model/config.json',
            ckpt_path='assets/tts_model/checkpoint.pth')
speaker_ids = tts_model.hps.data.spk2id
speak_id = speaker_ids['EN-Newest']

# Load a demo data, where demo.wav is the training data kzrptf.wav, and 
# its expected_text is "could we swim in the pool all day today instead of working in the yard"
data_audio, sampling_rate = librosa.load('demo.wav', sr=16000)
data_text = 'could we swim in the pool all day today instead of working in the yard'
data_task = 'sentence_repetition'
data_grade = '2'

# Next, the first type of localization is carried out: block TTS. Assume that the following four text blocks are phrases/words of interest.
data_text_blocks = [
    'could we swim',
    'in the pool all day today',
    'instead',
    'of working in the yard'
]
# Perform TTS on them separately
data_tts_blocks = []
for i, text in enumerate(data_text_blocks):
    tts_model.tts_to_file(text, speak_id, f'demo_tts_block{i}.wav', speed=0.7, quiet=True)
    data_tts_blocks.append(librosa.load(f'demo_tts_block{i}.wav', sr=16000)[0])

# Then, compare them separately to complete the first type of localization.
audio_tokens, task_id, grade_id = processor_func(data_audio, data_tts_blocks, data_task, data_grade)
with torch.no_grad():
    for k, v in audio_tokens.items():
        audio_tokens[k] = v.to(CFG.device).to(dtype)
    task_id = task_id.to(CFG.device).to(dtype)
    grade_id = task_id.to(CFG.device).to(dtype)
    preds = model(audio_tokens, audio_tokens, task_id, grade_id).cpu().numpy()
# The preds_dict here is the matching degree between the demo speech and each text block, 
# that is, the performance of the demo speech on each text block.
preds_dict = {text:pred for text, pred in zip(data_text_blocks, preds)}
print(preds_dict)
# This method can perform localization on the text level, but it is difficult to locate speech, and it cannot distinguish repeated text.

print('')

# We then do the second type of localization: sliding window + block TTS. We continue to use the above block TTS configuration, 
# and perform sliding window on the speech on this basis.
# First, set the window length and stride length of this demo, which are set to 1s and 0.5s.
window_len = 1 * 16000
stride_len = 0.5 * 16000
preds_windows = []
# Compare each window separately to complete the second type of localization.
n_windows = int(np.ceil((len(data_audio)+stride_len-window_len)/stride_len))
for i in range(n_windows):
    starti = round(i*stride_len)
    endi = round(starti+window_len)
    audio_block = data_audio[starti:endi]
    audio_tokens, task_id, grade_id = processor_func(audio_block, data_tts_blocks, data_task, data_grade)
    for k, v in audio_tokens.items():
        audio_tokens[k] = v.to(CFG.device).to(dtype)
    task_id = task_id.to(CFG.device).to(dtype)
    grade_id = task_id.to(CFG.device).to(dtype)
    with torch.no_grad():
        preds = model(audio_tokens, audio_tokens, task_id, grade_id).cpu().numpy()
    preds_windows.append(preds)
# The preds_dict here is the matching degree between each window of the demo speech and each text block, 
# that is, the performance of each location of the demo speech corresponding to each location of the text.
for i, preds in enumerate(preds_windows):
    preds_dict = {text:pred for text, pred in zip(data_text_blocks, preds)}
    print(f'window {i+1}: ', preds_dict)
# This method can locate text and speech, and can also distinguish repeated text.