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

from transformers import WhisperProcessor, WhisperConfig, GenerationConfig
from transformers import WhisperForConditionalGeneration

from utils import TranscriptionDatasetInfer
from utils import clean
from utils import get_max_length

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    """
    Formal wrapper around main script logic.
    """

    parser = ArgumentParser()
    parser.add_argument('--input_dir', default='train', type=str, help='Directory with a training dataset')
    parser.add_argument('--model_dir', default='tuned_transcriber_0_mediumen', type=str, help='Directory with trained models')
    parser.add_argument('--output_file', default='transcriptions_0_mediumen.csv', type=str, help='Output file name')
    parser.add_argument('--image_encoder_name', default='openai/whisper-medium.en', type=str, help='Audio encoder architecture')
    parser.add_argument('--max_length_whisper', default=-1, type=int, help='Maximum length of text for Whisper (tokenization and generation), -1 for auto')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int, help='Number of workers')
    parser.add_argument('--use_amp', default=1, type=int, choices=[0, 1], help='Whether to use auto mixed precision')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')
    args = parser.parse_args()
    for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

    device = torch.device(args.device)

    train_df = pd.read_csv(os.path.join(args.input_dir, 'train_labels.csv'))
    train_meta_df = pd.read_csv(os.path.join(args.input_dir, 'train_metadata.csv'))
    train_df = pd.merge(train_meta_df, train_df, on='filename', how='left')
    train_df['file'] = train_df['filename'].map(lambda x: os.path.join(args.input_dir, x))
    print('N examples:', len(train_df))

    # Create transcription column
    train_df['trans'] = 'empty transcription'

    # Split
    train_df['fold_id'] = 0
    train_df = train_df.reset_index(drop=True)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
    for fold_id, (train_index, val_index) in enumerate(kf.split(train_df, train_df['score'].values)):
        train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
    train_df = train_df.sample(frac=1.0, random_state=34)
    train_df = train_df.reset_index(drop=True)

    processor = WhisperProcessor.from_pretrained(args.image_encoder_name)
    generation_config = GenerationConfig.from_pretrained(args.image_encoder_name)
    config = WhisperConfig.from_pretrained(args.image_encoder_name)
    model = WhisperForConditionalGeneration(config).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Compute maximum length for Whisper if not specified
    if args.max_length_whisper == -1 or args.max_length_whisper is None:
        max_length_whisper = get_max_length(train_df['expected_text'].unique(), processor.tokenizer)
    else:
        max_length_whisper = args.max_length_whisper

    #--------------------------------------------------------------------------

    for fold_id in range(0, 5):
        print('Fold:', fold_id)
        tr_df = train_df[train_df['fold_id'] != fold_id]
        val_df = train_df[train_df['fold_id'] == fold_id]

        print('Init datasets...')
        val_dataset = TranscriptionDatasetInfer(val_df, proc_image=processor)
        val_loader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            pin_memory=True,)


        weight = glob.glob(os.path.join(args.model_dir, 'model-f%d-*.bin' % fold_id))[0]
        model.load_state_dict(torch.load(weight, map_location=device))
        print('Loaded model:', weight)

        results = []
        for batch_id, batch in enumerate(val_loader):
            input_features = batch['image'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                if generation_config.is_multilingual:
                    # When model is initialized from config, we need to pass
                    # "generation_config" explicitly in order to use "language" parameter.
                    predicted_ids = model.generate(input_features, max_length=max_length_whisper, 
                                                   generation_config=generation_config, language='English')
                else:
                    predicted_ids = model.generate(input_features, max_length=max_length_whisper)
            predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            results.extend(predicted_text)
            print(batch_id, end='\r')

        train_df.loc[train_df['fold_id'] == fold_id, 'trans'] = results

    #--------------------------------------------------------------------------

    # Clean
    train_df['trans'] = train_df['trans'].fillna('empty transcription')
    train_df.loc[train_df['trans'] == '', 'trans'] = 'empty transcription'
    train_df['trans_raw'] = train_df['trans']
    train_df['trans'] = train_df['trans'].map(clean)

    train_df['expected_text'] = train_df['expected_text'].fillna('empty text')
    train_df.loc[train_df['expected_text'] == '', 'expected_text'] = 'empty text'
    train_df['expected_text'] = train_df['expected_text'].map(clean)

    # Save
    train_df.to_csv(args.output_file, index=False)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

