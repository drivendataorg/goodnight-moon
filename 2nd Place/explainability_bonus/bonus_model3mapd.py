"""
This code shows the model training and validation processes of the cosine similarity solution, 
and its model structure is Figure 4 in Bonus Round Write-up.pdf.
"""

# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# ====================================================
# CFG
# ====================================================
# Here are the various configurations and hyperparameters of the model. You can see that I use whisper-medium as the speech model.
# I tried wav2vec2 and other speech/audio models. In theory, wav2vec2 for speech recognition is also a good speech feature extraction model,
# but its performance was much worse than whisper's. 
class CFG:
    debug=False
    apex=True
    print_freq=20
    num_workers=8
    speech_model = "openai/whisper-medium"
    scheduler='cosine' # ['linear', 'cosine']
    data_path = 'data/train_metadata.csv'
    label_path = 'data/train_labels.csv'
    audio_folder = 'data/train_audio/'
    label_audio_folder = 'tts_data/'
    refer_pred_folder = 'refer_preds/'
    model_save_name = 'model3mapd'
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    scheduler_epochs=4
    epochs = 4
    encoder_lr=1e-5
    decoder_lr=1e-5
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=16
    max_speech_len = 16
    weight_decay=1e-3
    data_filter_rate = 0.0
    speech_dropout = 0.0
    gradient_accumulation_steps=1
    max_grad_norm=20000
    seed=88
    data_seed = 88
    train=True
    gradient_checkpointing = True
    target_cols = ['score']
    
# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from glob import glob

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import argparse
import gc
import copy

import tokenizers
import transformers
import torchaudio
import librosa
import pyworld as pw
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--n_folds', type=int, default=4)
args = parser.parse_args()

CFG.seed = CFG.seed + args.fold
device = 'cuda'
CFG.n_fold = args.n_folds
CFG.trn_fold=[args.fold]

data_df = pd.read_csv(CFG.data_path)
label_df = pd.read_csv(CFG.label_path)
data_df = data_df.merge(label_df, how='inner', on='filename')

# Set the data allocation for cross-validation (cv)
data_df = data_df.sample(frac=1, random_state=CFG.data_seed+2).reset_index(drop=True)

skf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.data_seed)
for i, (_, val_index) in enumerate(skf.split(data_df)):
    data_df.loc[val_index, "fold"] = i

# ====================================================
# Utils
# ====================================================
def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.seed)

train = data_df

# ====================================================
# tokenizer
# ====================================================
# Here I set chunk_length to CFG.max_speech_len (16), which avoids many redundant pads affecting the running speed.
CFG.processor = AutoProcessor.from_pretrained(CFG.speech_model, chunk_length=CFG.max_speech_len)

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

# Here is the data augmentation part, which consists of 6 parts: random cropping, random pitch shift, 
# random time stretching, random noise, random volume, and random spectrum envelope.
def augment_audio(audio, sample_rate, full=True):
    """
    Apply data augmentation to the given audio
    """
    # Cut
    if np.random.rand() < 0.4:
        olen = len(audio)
        for _ in range(np.random.randint(1, 7)):
            cut_length = int(np.random.uniform(0.025, 0.075)*len(audio))
            start_idx = np.random.randint(0, cut_length)
            audio[start_idx:start_idx + cut_length] = 0
    
    # Apply pitch shift
    if full and np.random.rand() < 0.4:
        pitch_shift = np.random.uniform(-5, 5)  # Random pitch shift in the specified range
        audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
    
    # Apply time stretching
    if np.random.rand() < 0.4:
        stretch_factor = np.random.uniform(0.7, 1.3)  # Random time stretch factor
        audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    # Add random noise
    if np.random.rand() < 0.4:
        noise = np.random.randn(len(audio)) * 0.03  # Generate random noise
        audio = audio + noise  # Add noise to the original signal

    # Change volume
    if np.random.rand() < 0.4:
        factor = np.random.uniform(0.7, 1.3)
        audio = audio * factor

    # Change envelope
    if full and np.random.rand() < 0.3:
        noise_range = [-0.3, 0.5]
        f0, sp, ap = pw.wav2world(audio.astype(np.float64), sample_rate)
        sp_noisy = sp + (noise_range[1]-noise_range[0])*np.random.randn(*sp.shape) + noise_range[0]
        y_augmented = pw.synthesize(f0, sp_noisy, ap, sample_rate)
        
    return audio

# The output audio token of this dataset is a combination of TTS speech and real speech. 
# When looking for different solutions, I also used other TTS models to analyze the impact of different TTS models 
# on the results. I ended up choosing the best performing model: MeloTTS's EN-Newest.
# There are two audio_tokens outputs for being compatible with the structure of processing two speeches separately.
class TrainDataset(Dataset):
    def __init__(self, data_df, is_train=False):
        self.audio_files = data_df['filename'].values
        self.task_ids = [CFG.task2id[task] for task in data_df['task'].values]
        self.text_array =  data_df['expected_text'].values
        self.grade_ids = [CFG.grade2id[grade] for grade in data_df['grade'].values]
        self.labels = data_df['score'].values
        self.is_train = is_train

        data_paths = pd.read_csv(CFG.label_audio_folder+'data_paths.csv')[['text', 'path']].values
        self.label_audio_dict = {text:path for text, path in data_paths}
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, item):
        audio_file = self.audio_files[item]
        raw_text = self.text_array[item]
        task_id = self.task_ids[item]
        grade_id = self.grade_ids[item]
        label = self.labels[item]
        
        audio, sampling_rate = librosa.load(CFG.audio_folder+audio_file, sr=16000)
        audio2, sampling_rate2 = librosa.load(self.label_audio_dict[raw_text], sr=16000)
        if self.is_train:
            audio = augment_audio(audio, sampling_rate)
            audio2 = augment_audio(audio2, sampling_rate, full=False)
        audio = np.concatenate([audio2, audio], axis=-1)
        audio = torch.tensor(audio)
        audio_tokens = CFG.processor.feature_extractor(audio, sampling_rate=16000)
        audio_tokens['input_features'] = audio_tokens['input_features'][0]

        # Two additional features are added here: task and grade. In my experiment, they have no obvious effect on the results, 
        # but will slightly increase the stability of the results. These two features are not used in the cosine similarity solution.
        task_id = torch.tensor(task_id) / len(CFG.task2id)
        grade_id = torch.tensor(grade_id) / len(CFG.grade2id)

        label = torch.tensor(label).to(torch.long)
        return audio_tokens, audio_tokens, task_id, grade_id, label

def collate(batch):     
    audio_tokens = CFG.processor.feature_extractor.pad([data[0] for data in batch], padding='longest',
                                                      return_tensors="pt")
    label_audio_tokens = CFG.processor.feature_extractor.pad([data[1] for data in batch], padding='longest',
                                                      return_tensors="pt")
    task_ids = torch.stack([data[2] for data in batch], dim=0)
    grade_ids = torch.stack([data[3] for data in batch], dim=0)
    labels = torch.stack([data[4] for data in batch], dim=0)
    return audio_tokens, label_audio_tokens, task_ids, grade_ids, labels.to(torch.float)

def worker_init_fn(worker_id):
    np.random.seed(CFG.seed+worker_id*CFG.num_workers)

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

# This is the attention pooling part.
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

# The basic structure flow of the model is: Whisper is used to extract 2 speech feature sequences, then attention pooling is used to generate 2 speech features,
# and then the cosine similarity of the two speech features is calculated to get the final output.
# The model of the competition is a binary classification model. For better explainability, the cosine similarity solution is used here for illustration.
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(CFG.speech_model)
        config.max_source_positions = CFG.max_speech_len * 50
        self.speech_model = AutoModel.from_pretrained(CFG.speech_model, config=config, 
                                                      ignore_mismatched_sizes=True).to(device)        
        # Only the encoder is needed, so the decoder is deleted to reduce the model size.
        del self.speech_model.decoder 
        if CFG.gradient_checkpointing:
            self.speech_model.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(CFG.speech_dropout)
        self.pool = AttentionPooling(self.speech_model.config.hidden_size)
        
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
    def feature(self, audio_tokens, label_audio_tokens):
        audio_features = self.speech_model._mask_input_features(audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        audio_embedding = self.dropout(label_audio_embedding)
        audio_embedding = self.pool(audio_features)
        
        audio_features = self.speech_model._mask_input_features(label_audio_tokens['input_features'])
        audio_features = self.speech_model.encoder(audio_features)[0]
        label_audio_embedding = self.dropout(label_audio_embedding)
        label_audio_embedding = self.pool(audio_features)
        
        
        return audio_embedding, label_audio_embedding

    def forward(self, audio_tokens, label_audio_tokens, task_ids, grade_ids):
        audio_embedding, label_audio_embedding = self.feature(audio_tokens, label_audio_tokens)
        output = self.cos(audio_embedding, label_audio_embedding).to(torch.float32)
        return output
    
# ====================================================
# Loss
# ====================================================
import torch
from torch import nn

from sklearn.metrics import log_loss
get_score = log_loss
LOSS_FN = torch.nn.BCELoss()
LOSS_FN_NOREDUCE = torch.nn.BCELoss(reduction='none')

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


# Training function
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    # Because mixed precision training is enabled, the gradient scaling that matches it is used.
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (audio_tokens, label_audio_tokens, task_ids, grade_ids, labels) in enumerate(train_loader):
        for k, v in audio_tokens.items():
            audio_tokens[k] = v.to(device)
        for k, v in label_audio_tokens.items():
            label_audio_tokens[k] = v.to(device)
        task_ids = task_ids.to(device)
        grade_ids = grade_ids.to(device)
        labels = labels.to(device)
        
        batch_size = labels.size(0)
        # Enabled mixed precision training
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(audio_tokens, label_audio_tokens, task_ids, grade_ids)
        loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # The gradient clipping here improves the final performance of the model
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            LOGGER.info(f'Fold {fold} '
                  'Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        gc.collect()
        torch.cuda.empty_cache()
    return losses.avg

# Validation function
def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (audio_tokens, label_audio_tokens, task_ids, grade_ids, labels) in enumerate(valid_loader):
        for k, v in audio_tokens.items():
            audio_tokens[k] = v.to(device)
        for k, v in label_audio_tokens.items():
            label_audio_tokens[k] = v.to(device)
        task_ids = task_ids.to(device)
        grade_ids = grade_ids.to(device)
        labels = labels.to(device)
        
        batch_size = labels.size(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                y_preds = model(audio_tokens, label_audio_tokens, task_ids, grade_ids)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            LOGGER.info(f'Fold {fold} '
                  'EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
        gc.collect()
        torch.cuda.empty_cache()
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold']!=fold]
    print('train data:', len(train_folds))
    if CFG.data_filter_rate > 0:
        train_folds = train_folds[train_folds['loss']<CFG.data_filter_thr].reset_index(drop=True)
        print('train data after:', len(train_folds))
    else:
        train_folds = train_folds.reset_index(drop=True)
    train_dataset = TrainDataset(train_folds, is_train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size, worker_init_fn=worker_init_fn,
                              shuffle=True, collate_fn=collate,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    
    
    valid_folds = folds[folds['fold']==fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    valid_dataset = TrainDataset(valid_folds)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False, collate_fn=collate,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel()
    
    # Here sets which parts of the model do not need weight decay (regularization)
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.speech_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.speech_model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "_model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    # The cosine scheduler is used, which is the best choice based on experience.
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.scheduler_epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = LOSS_FN
    
    best_score = np.inf
    for epoch in range(CFG.epochs):
        start_time = time.time()
        model.to(device)
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score = get_score(valid_labels, predictions)
        elapsed = time.time() - start_time

        LOGGER.info(f'Fold {fold} Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Fold {fold} Epoch {epoch+1} - Score: {score:.4f}')
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Fold {fold} Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            save_file = CFG.model_save_name
            model_to_save = copy.deepcopy(model.cpu())
            # The model will eventually output the cv performance directly to facilitate the analysis of the advantages and disadvantages of the model.
            # The score (label), prediction and loss corresponding to each sample in the cv validation part of each fold will be saved 
            # in the model file and saved separately as a csv file.
            # You can run check_cv_details.py to output the cv details of each fold of each model.
            # According to the analysis of cv, the task my model is least good at is nonword_repetition (cv 0.31-0.35), 
            # and the task my model is best at is blending (cv 0.13-0.17); the grade my model is least good at is 1 
            # (cv 0.24-0.27), and the grade my model is best at is 3 (0.14-0.22).
            torch.save({
                'filename': valid_folds['filename'].values, 
                'model': model_to_save.to(torch.float16),
                'score': valid_labels[:, 0], 
                'pred': predictions
            },OUTPUT_DIR+save_file+f'_{fold}.pth')

            pred_loss = LOSS_FN_NOREDUCE(torch.tensor(predictions).to(torch.float32), 
                                         torch.tensor(valid_labels[:, 0]).to(torch.float32)).numpy()
            pd.DataFrame({
                'filename': valid_folds['filename'].values, 
                'score': valid_labels[:, 0], 
                'pred': predictions,
                'loss': pred_loss
            }).to_csv(f'fold{fold}_best_preds.csv', index=False)

    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == '__main__':
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold)