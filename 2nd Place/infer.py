import pickle

# ====================================================
# CFG
# ====================================================
used_models = [
        'assets/model3mapd_0.pth',
        'assets/model3mapd_1.pth',
        'assets/model3mapd_2.pth',
        
        'assets/model3mcp_1.pth',
        'assets/model3mcp_2.pth',
        'assets/model3mcp_3.pth',

        'assets/model3mb_2.pth',
        'assets/model3mb_3.pth',
        'assets/model3mb_4.pth',

        'assets/model2m_3.pth',
        'assets/model2m_4.pth',
        'assets/model2m_5.pth',

        'assets/model2dm_4.pth',
        'assets/model2dm_5.pth',
        'assets/model2dm_0.pth',

        'assets/model1m_5.pth',
        'assets/model1m_0.pth',
        'assets/model1m_1.pth',


        'assets/model2ds_0.pth',
        'assets/model2ds_1.pth',
        'assets/model2ds_2.pth',

        'assets/model2s_3.pth',
        'assets/model2s_4.pth',
        'assets/model2s_5.pth',

        'assets/model3scp_0.pth',
        'assets/model3scp_1.pth',
        'assets/model3scp_2.pth',

        'assets/model3dsap_3.pth',
        'assets/model3dsap_4.pth',
        'assets/model3dsap_5.pth',
]


with open('model_configs.pkl', 'rb') as f:
    model_configs = pickle.load(f)

new_model_configs = {}
for mconfig in model_configs:
    weight = mconfig['weight']
    new_paths = []
    for path in mconfig['paths']:
        if path in used_models:
            new_paths.append(path)
    if len(new_paths) > 0:
        mconfig['weight'] = float(weight*len(new_paths)/len(mconfig['paths']))
        mconfig['paths'] = new_paths
        if (mconfig['speech_processor'], mconfig['dataset_type']) in new_model_configs:
            new_model_configs[(mconfig['speech_processor'], mconfig['dataset_type'])].append(mconfig)
        else:
            new_model_configs[(mconfig['speech_processor'], mconfig['dataset_type'])] = [mconfig]
model_configs = new_model_configs
print('model_configs: ', model_configs)

class CFG:
    num_workers=16
    data_path = 'data/test_metadata.csv'
    audio_folder = 'data/'
    model_configs = model_configs

    seed=42
    target_cols = ['score']
    device = 'cuda'
    
# ====================================================
# Library
# ====================================================
import gc
import re
import os
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

import transformers
import torchaudio
import librosa
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.seed)

# ====================================================
# Dataset
# ====================================================
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

from melo.api import TTS
def get_tts_audio(metadata):
    model = TTS(language='EN_NEWEST', device=CFG.device, use_hf=False,
                config_path='assets/tts_model/config.json',
                ckpt_path='assets/tts_model/checkpoint.pth')
    speaker_ids = model.hps.data.spk2id
    speak_id = speaker_ids['EN-Newest']
    
    train_metadata = pd.read_csv(CFG.data_path)
    texts = np.array(metadata['expected_text'].unique())
    paths = []
    os.system('mkdir tts_data_test')
    for i, expected_text in tqdm(enumerate(texts)):
        speed = 0.7
        save_path1 = f"tts_data_test/{i}.wav"
        model.tts_to_file(expected_text, speak_id, save_path1, speed=speed, quiet=True)
        paths.append(save_path1)
    pd.DataFrame({'path':paths, 'text': texts}).to_csv('tts_data_test/data_paths.csv', index=False)

class TestDataset_base(Dataset):
    def __init__(self, data_df):
        self.audio_files = data_df['filename'].values
        self.task_ids = [CFG.task2id[task] for task in data_df['task'].values]
        self.text_array =  data_df['expected_text'].values
        self.grade_ids = [CFG.grade2id[grade] for grade in data_df['grade'].values]

        data_paths = pd.read_csv('tts_data_test/data_paths.csv')[['text', 'path']].values
        self.label_audio_dict = {text:path for text, path in data_paths}
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, item):
        audio_file = self.audio_files[item]
        raw_text = self.text_array[item]
        task_id = self.task_ids[item]
        grade_id = self.grade_ids[item]
        
        audio, sampling_rate = librosa.load(CFG.audio_folder+audio_file)
        audio2, sampling_rate2 = librosa.load(self.label_audio_dict[raw_text])
        audio = np.concatenate([audio2, audio], axis=-1)
        
        audio = torch.tensor(audio)
        audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=16000)
        audio_tokens = CFG.processor.feature_extractor(audio, sampling_rate=16000)
        audio_tokens['input_features'] = audio_tokens['input_features'][0]

        task_id = torch.tensor(task_id) / len(CFG.task2id)
        grade_id = torch.tensor(grade_id) / len(CFG.grade2id)
        return audio_tokens, audio_tokens, task_id, grade_id

class TestDataset_fixsr(Dataset):
    def __init__(self, data_df):
        self.audio_files = data_df['filename'].values
        self.task_ids = [CFG.task2id[task] for task in data_df['task'].values]
        self.text_array =  data_df['expected_text'].values
        self.grade_ids = [CFG.grade2id[grade] for grade in data_df['grade'].values]

        data_paths = pd.read_csv('tts_data_test/data_paths.csv')[['text', 'path']].values
        self.label_audio_dict = {text:path for text, path in data_paths}
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, item):
        audio_file = self.audio_files[item]
        raw_text = self.text_array[item]
        task_id = self.task_ids[item]
        grade_id = self.grade_ids[item]
        
        audio, sampling_rate = librosa.load(CFG.audio_folder+audio_file, sr=16000)
        audio2, sampling_rate2 = librosa.load(self.label_audio_dict[raw_text], sr=16000)

        audio = np.concatenate([audio2, audio], axis=-1)
        audio = torch.tensor(audio)
        audio_tokens = CFG.processor.feature_extractor(audio, sampling_rate=16000)
        audio_tokens['input_features'] = audio_tokens['input_features'][0]

        task_id = torch.tensor(task_id) / len(CFG.task2id)
        grade_id = torch.tensor(grade_id) / len(CFG.grade2id)
        return audio_tokens, audio_tokens, task_id, grade_id

TestDatasets = {'base': TestDataset_base, 'fixsr': TestDataset_fixsr}

def collate(batch):     
    audio_tokens = CFG.processor.feature_extractor.pad([data[0] for data in batch], padding='longest',
                                                      return_tensors="pt")
    label_audio_tokens = CFG.processor.feature_extractor.pad([data[1] for data in batch], padding='longest',
                                                      return_tensors="pt")
    task_ids = torch.stack([data[2] for data in batch], dim=0)
    grade_ids = torch.stack([data[3] for data in batch], dim=0)
    return audio_tokens, label_audio_tokens, task_ids, grade_ids

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask=None):
        if attention_mask is None:
            mean_embeddings = torch.mean(last_hidden_state, 1)
            return mean_embeddings
            
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
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

class CustomModel_base(nn.Module):
    def feature(self, audio_tokens, label_audio_tokens):
        audio_features = self.speech_model._mask_input_features(label_audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        audio_features = self.dropout(audio_features)
        label_audio_embedding = self.pool(audio_features)
        
        return label_audio_embedding

    def forward(self, audio_tokens, label_audio_tokens, task_ids, grade_ids):
        label_audio_embedding = self.feature(audio_tokens, label_audio_tokens)
        other_features = torch.stack([task_ids, grade_ids], dim=-1)
        label_audio_embedding = torch.cat([label_audio_embedding, other_features], dim=-1)
        output = self.fc1(label_audio_embedding).sigmoid()[:, 0].to(torch.float32)
        return output

class CustomModel_base2(nn.Module):
    def feature(self, audio_tokens, label_audio_tokens):
        audio_features = self.speech_model._mask_input_features(label_audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        audio_features = self.dropout(audio_features)
        label_audio_embedding = self.pool(audio_features)
        
        return label_audio_embedding

    def forward(self, audio_tokens, label_audio_tokens, task_ids, grade_ids):
        label_audio_embedding = self.feature(audio_tokens, label_audio_tokens)
        other_features = torch.stack([task_ids, grade_ids], dim=-1)
        label_audio_embedding = torch.cat([label_audio_embedding, other_features], dim=-1)
        output = self.fc1(label_audio_embedding)
        output = self.fc2(output).sigmoid()[:, 0].to(torch.float32)
        return output

class CustomModel_combpoolsv2(nn.Module):
    def feature(self, audio_tokens, label_audio_tokens):
        audio_features = self.speech_model._mask_input_features(label_audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        audio_features = self.dropout(audio_features)
        label_audio_embedding = torch.cat([self.pool1(audio_features), 
                                           self.pool2(audio_features)], dim=-1)
        
        return label_audio_embedding

    def forward(self, audio_tokens, label_audio_tokens, task_ids, grade_ids):
        label_audio_embedding = self.feature(audio_tokens, label_audio_tokens)
        other_features = torch.stack([task_ids, grade_ids], dim=-1)
        label_audio_embedding = torch.cat([label_audio_embedding, other_features], dim=-1)
        output = self.fc1(label_audio_embedding)
        output = self.fc2(output).sigmoid()[:, 0].to(torch.float32)
        return output

class CustomModel_attpool(nn.Module):
    def feature(self, audio_tokens, label_audio_tokens):
        audio_features = self.speech_model._mask_input_features(label_audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        audio_features = self.dropout(audio_features)
        label_audio_embedding = self.pool(audio_features)
        
        return label_audio_embedding

    def forward(self, audio_tokens, label_audio_tokens, task_ids, grade_ids):
        label_audio_embedding = self.feature(audio_tokens, label_audio_tokens)
        other_features = torch.stack([task_ids, grade_ids], dim=-1)
        label_audio_embedding = torch.cat([label_audio_embedding, other_features], dim=-1)
        output = self.fc1(label_audio_embedding).sigmoid()[:, 0].to(torch.float32)
        return output

class CustomModel_attpooldov3(nn.Module):
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
        
CustomModels = {
    'base': CustomModel_base, 'base2': CustomModel_base2, 'combpoolsv2': CustomModel_combpoolsv2, 
    'attpool': CustomModel_attpool, 'attpooldov3': CustomModel_attpooldov3
}

# ====================================================
# Helper functions
# ====================================================
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def test_fn(test_loader, models_list, device):
    for models in models_list:
        for model in models:
            model.eval()
    for k, v in model.named_parameters():
        dtype = v.dtype
        break
        
    preds = [[] for _ in models_list]
    for step, (audio_tokens, label_audio_tokens, task_ids, grade_ids) in enumerate(test_loader):
        for k, v in audio_tokens.items():
            audio_tokens[k] = v.to(device).to(dtype)
        for k, v in label_audio_tokens.items():
            label_audio_tokens[k] = v.to(device).to(dtype)
        task_ids = task_ids.to(device).to(dtype)
        grade_ids = grade_ids.to(device).to(dtype)
        
        with torch.no_grad():
            #with torch.cuda.amp.autocast(enabled=CFG.apex):
            for mi, models in enumerate(models_list):
                y_preds = 0
                for model in models:
                    y_preds += model(audio_tokens, label_audio_tokens, task_ids, grade_ids)
                y_preds /= len(models)
                preds[mi].append(y_preds.to('cpu').numpy())

        gc.collect()
        torch.cuda.empty_cache()

    for i in range(len(preds)):
        preds[i] = np.concatenate(preds[i])
    return preds
    
if __name__ == '__main__':
    data_df = pd.read_csv(CFG.data_path)
    get_tts_audio(data_df)
    gc.collect()
    torch.cuda.empty_cache()
    
    preds = 0
    weights = 0
    for speech_processor, dataset_type in CFG.model_configs:
        model_config_list = CFG.model_configs[(speech_processor, dataset_type)]

        models_list = []
        for model_config in model_config_list:
            models = []
            CustomModel = CustomModels[model_config['model_type']]
            for path in model_config['paths']:
                model_cache = torch.load(path)
                models.append(model_cache['model'].to(CFG.device))
            models_list.append(models)
        CFG.max_speech_len = model_config['max_speech_len']
        CFG.processor = AutoProcessor.from_pretrained(model_config['speech_processor'], 
                                                      chunk_length=CFG.max_speech_len)
        test_dataset = TestDatasets[model_config['dataset_type']](data_df)
        test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'],
                                  shuffle=False, collate_fn=collate,
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        preds_cache = test_fn(test_loader, models_list, CFG.device)
        for pc, model_config in zip(preds_cache, model_config_list):
            preds += pc * model_config['weight']
            weights += model_config['weight']
        
        del model_cache, models, preds_cache, models_list
        gc.collect()
        torch.cuda.empty_cache()    
    preds /= weights
    submission = pd.DataFrame({'filename': data_df['filename'], 'score': preds})
    submission.to_csv('submission.csv', index=False)