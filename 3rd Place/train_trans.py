#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from argparse import ArgumentParser

from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
import evaluate

from utils import TranscriptionDatasetTrain
from utils import clean
from utils import keep_n_last_ckpt
from utils import get_max_length

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input_dir', default='train', type=str, help='Directory with a training dataset')
parser.add_argument('--output_dir', default='tuned_transcriber_0_mediumen', type=str, help='Directory to save model checkpoints')
parser.add_argument('--image_encoder_name', default='openai/whisper-medium.en', type=str, help='Audio encoder architecture')
parser.add_argument('--max_length_whisper', default=-1, type=int, help='Maximum length of text for Whisper (tokenization and generation), -1 for auto')
parser.add_argument('--n_epochs', default=4, type=int, help='Number of epochs to train')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--accum', default=9, type=int, help='Number of steps for gradient accumulation')
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
parser.add_argument('--num_workers', default=os.cpu_count(), type=int, help='Number of workers')
parser.add_argument('--use_amp', default=1, type=int, choices=[0, 1], help='Whether to use auto mixed precision')
parser.add_argument('--initial_fold', default=0, type=int, help='Initial fold index (0 to 4)')
parser.add_argument('--final_fold', default=1, type=int, help='Final fold index (1 to 5)')
parser.add_argument('--reduce_p', default=1, type=int, help='Patience for learning rate reduction')
parser.add_argument('--reduce_f', default=0.5, type=float, help='Factor for learning rate reduction')
parser.add_argument('--reduce_mode', default='min', type=str, help='Mode (min/max) for learning rate reduction')
parser.add_argument('--device', default='cuda:0', type=str, help='Device')
args = parser.parse_args()
for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
# Train/val split
#------------------------------------------------------------------------------

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device)

train_df = pd.read_csv(os.path.join(args.input_dir, 'train_labels.csv'))
train_meta_df = pd.read_csv(os.path.join(args.input_dir, 'train_metadata.csv'))
train_df = pd.merge(train_meta_df, train_df, on='filename', how='left')
train_df['file'] = train_df['filename'].map(lambda x: os.path.join(args.input_dir, x))
print('N examples:', len(train_df))

# Clean
train_df['expected_text'] = train_df['expected_text'].fillna('empty text')
train_df.loc[train_df['expected_text'] == '', 'expected_text'] = 'empty text'
train_df['expected_text'] = train_df['expected_text'].map(clean)

# Split
train_df['fold_id'] = 0
train_df = train_df.reset_index(drop=True)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
for fold_id, (train_index, val_index) in enumerate(kf.split(train_df, train_df['score'].values)):
    train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
train_df = train_df.sample(frac=1.0, random_state=34)
train_df = train_df.reset_index(drop=True)

processor = WhisperProcessor.from_pretrained(args.image_encoder_name, language='English', task='transcribe')

# Compute maximum length for Whisper if not specified
if args.max_length_whisper == -1 or args.max_length_whisper is None:
    max_length_whisper = get_max_length(train_df['expected_text'].unique(), processor.tokenizer)
else:
    max_length_whisper = args.max_length_whisper

# Init metrics
wer_metric = evaluate.load('wer')
cer_metric = evaluate.load('cer')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for fold_id in range(args.initial_fold, args.final_fold):
    print('Fold:', fold_id)

    tr_df = train_df[train_df['fold_id'] != fold_id]
    val_df = train_df[train_df['fold_id'] == fold_id]

    # Select only positive examples
    tr_df = tr_df[tr_df['score'] == 1]
    val_df = val_df[val_df['score'] == 1]
    
    print('Train:', tr_df.shape)
    print('Val:', val_df.shape)

    print('Init datasets...')
    train_dataset = TranscriptionDatasetTrain(tr_df, proc_image=processor, max_length=max_length_whisper)
    val_dataset = TranscriptionDatasetTrain(val_df, proc_image=processor, max_length=max_length_whisper)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                        pin_memory=True,)
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False,
                        pin_memory=True,)

    print('Init model...')
    model = WhisperForConditionalGeneration.from_pretrained(args.image_encoder_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=args.reduce_mode, factor=args.reduce_f, 
                    patience=args.reduce_p, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp))
    best_score = 100
    
    print('Start training...')
    for epoch_id in range(args.n_epochs):
        print('Epoch: %d' % epoch_id)
        start = time.time()
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
        avg_loss = 0

        for batch_id, batch in enumerate(train_loader):
            x = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                input_ids[input_ids == processor.tokenizer.pad_token_id] = -100
                loss = model(input_features=x, labels=input_ids).loss
                loss = loss / args.accum
            avg_loss += (loss.item() * args.accum) / len(train_loader)

            print('Batch: %04d    Loss: %.4f    Time: %d' % 
                  (batch_id, avg_loss, (time.time() - start)), end='\r')

            scaler.scale(loss).backward()
            if (batch_id+1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print('\nEval: %d' % epoch_id)
        model.eval()
        torch.set_grad_enabled(False)
        avg_loss = 0
        all_trues = []
        all_preds = []

        for batch_id, batch in enumerate(val_loader):
            x = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                trues = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                input_ids[input_ids == processor.tokenizer.pad_token_id] = -100
                loss = model(input_features=x, labels=input_ids).loss
    
                predicted_ids = model.generate(x, max_length=max_length_whisper)
                preds = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

                all_trues.extend(trues)
                all_preds.extend(preds)

            avg_loss += loss.item() / len(val_loader)

            print('Val batch: %04d    Val loss: %.4f    Time: %d' % 
                  (batch_id, avg_loss, (time.time() - start)), end='\r')

        # Compute metrics
        wer_score = wer_metric.compute(predictions=all_preds, references=all_trues)
        cer_score = cer_metric.compute(predictions=all_preds, references=all_trues)

        # Update LR
        scheduler.step(avg_loss)

        # Save if loss improved
        if avg_loss < best_score:
            best_score = avg_loss            
            p = 'model-f%d-e%03d-%.4f-wer-%.4f-cer-%.4f.bin' % (fold_id, epoch_id, avg_loss, wer_score, cer_score)
            p = os.path.join(args.output_dir, p)
            torch.save(model.state_dict(), p)
            print('\nSaved model:', p)
        else:
            print('\nScore is not better: not saving the model')

        keep_n_last_ckpt(os.path.join(args.output_dir, 'model-f%d-*' % fold_id))
        print('Epoch time %d sec:' % (time.time() - start))


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
