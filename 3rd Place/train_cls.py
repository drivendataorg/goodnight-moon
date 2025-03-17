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
from transformers import AutoTokenizer, AutoModel

from utils import ClassificationDataset
from utils import MultimodalClassifier
from utils import clean
from utils import keep_n_best_ckpt
from utils import get_max_length

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input_dir', default='train', type=str, help='Directory with a training dataset')
parser.add_argument('--output_dir', default='tuned_cls_0_mediumen_rerun_0', type=str, help='Directory to save model checkpoints')
parser.add_argument('--image_encoder_name', default='openai/whisper-medium.en', type=str, help='Audio encoder architecture')
parser.add_argument('--text_encoder_name', default='microsoft/deberta-v3-base', type=str, help='Text encoder architecture')
parser.add_argument('--tuned_transcriber_dir', default='tuned_transcriber_0_mediumen', type=str, help='Directory with tuned transcription models')
parser.add_argument('--transcriptions', default='transcriptions_0_mediumen.csv', type=str, help='Dataframe with transcriptions')
parser.add_argument('--gen_voice_map', default='assets_new/gen_voice_map_fp16_seed33.pkl', type=str, help='Dictionary with a speech generated from corresponding expected text')
parser.add_argument('--max_length_deberta', default=-1, type=int, help='Maximum length of text for Deberta (tokenization), -1 for auto')
parser.add_argument('--n_epochs', default=2, type=int, help='Number of epochs to train')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--accum', default=4, type=int, help='Number of steps for gradient accumulation')
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

train_df = pd.read_csv(args.transcriptions)
train_df['file'] = train_df['filename'].map(lambda x: os.path.join(args.input_dir, x))

# Clean
train_df['trans'] = train_df['trans'].fillna('empty transcription')
train_df.loc[train_df['trans'] == '', 'trans'] = 'empty transcription'
train_df['trans'] = train_df['trans'].map(clean)
train_df['expected_text'] = train_df['expected_text'].fillna('empty text')
train_df.loc[train_df['expected_text'] == '', 'expected_text'] = 'empty text'
train_df['expected_text'] = train_df['expected_text'].map(clean)

processor = WhisperProcessor.from_pretrained(args.image_encoder_name)
tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)

# Compute maximum length for Deberta if not specified
if args.max_length_deberta == -1 or args.max_length_deberta is None:
    concat_list = train_df['expected_text'] + ' ' + train_df['trans']
    max_length_deberta = get_max_length(concat_list.unique(), tokenizer)
else:
    max_length_deberta = args.max_length_deberta

# Load pre-generated speech map
with open(args.gen_voice_map, 'rb') as f:
    gen_voice_map = pickle.load(f)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for fold_id in range(args.initial_fold, args.final_fold):
    print('Fold:', fold_id)

    tr_df = train_df[train_df['fold_id'] != fold_id]
    val_df = train_df[train_df['fold_id'] == fold_id]
    
    print('Train:', tr_df.shape)
    print('Val:', val_df.shape)
        
    print('Init datasets...')
    train_dataset = ClassificationDataset(
                        tr_df, gen_voice_map, proc_image=processor, 
                        proc_text=tokenizer, max_length=max_length_deberta)
    val_dataset = ClassificationDataset(
                        val_df, gen_voice_map, proc_image=processor, 
                        proc_text=tokenizer, max_length=max_length_deberta)
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
    model = MultimodalClassifier(
                image_encoder_name=args.image_encoder_name, 
                text_encoder_name=args.text_encoder_name, 
                tuned_transcriber=glob.glob(os.path.join(args.tuned_transcriber_dir, 'model-f%d-*' % fold_id))[0])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
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
        avg_acc = 0
        #
        for batch_id, batch in enumerate(train_loader):
            x = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                logits = model(x, input_ids, attention_mask)
                loss = criterion(logits, y)
                loss = loss / args.accum
            avg_loss += (loss.item() * args.accum) / len(train_loader)
            # acc
            with torch.no_grad():                
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()
                avg_acc += acc.item() / len(train_loader)

            print('Batch: %04d    Loss: %.4f    Acc: %.4f    Time: %d' % 
                  (batch_id, avg_loss, avg_acc, (time.time() - start)), end='\r')

            scaler.scale(loss).backward()
            if (batch_id+1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print('\nEval: %d' % epoch_id)
        model.eval()
        torch.set_grad_enabled(False)
        avg_loss = 0
        avg_acc = 0
        for batch_id, batch in enumerate(val_loader):
            x = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                logits = model(x, input_ids, attention_mask)
                loss = criterion(logits, y)
            avg_loss += loss.item() / len(val_loader)
            # acc
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()
                avg_acc += acc.item() / len(val_loader)

            print('Val batch: %04d    Val loss: %.4f    Val acc: %.4f    Time: %d' % 
                  (batch_id, avg_loss, avg_acc, (time.time() - start)), end='\r')

        # Update LR
        scheduler.step(avg_loss)

        # Save if loss improved
        if avg_loss < best_score:
            best_score = avg_loss            
            p = 'model-f%d-e%03d-%.4f-acc-%.4f.bin' % (fold_id, epoch_id, avg_loss, avg_acc)
            p = os.path.join(args.output_dir, p)
            torch.save(model.state_dict(), p)
            print('\nSaved model:', p)
        else:
            print('\nScore is not better: not saving the model')
        
        keep_n_best_ckpt(os.path.join(args.output_dir, 'model-f%d-*' % fold_id))
        print('Epoch time %d sec:' % (time.time() - start))


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
