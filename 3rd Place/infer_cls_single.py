#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser

from transformers import WhisperProcessor, WhisperConfig
from transformers import WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed

from utils import TranscriptionDatasetInfer
from utils import ClassificationDataset
from utils import MultimodalClassifier
from utils import clean
from utils import get_max_length

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    """
    Formal wrapper around main script logic.
    """

    parser = ArgumentParser()
    parser.add_argument('--input_dir', default='data', type=str, help='Directory with a test dataset')
    parser.add_argument('--assets_dir', default='assets', type=str, help='Directory with assets')
    parser.add_argument('--submission_file', default='submission.csv', type=str, help='Submission file path')
    parser.add_argument('--max_length_whisper', default=-1, type=int, help='Maximum length of text for Whisper (tokenization and generation), -1 for auto')
    parser.add_argument('--max_length_deberta', default=-1, type=int, help='Maximum length of text for Deberta (tokenization), -1 for auto')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int, help='Number of workers')
    parser.add_argument('--use_amp', default=1, type=int, choices=[0, 1], help='Whether to use auto mixed precision')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')
    args = parser.parse_args()
    for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

    device = torch.device(args.device)

    # Load and join test csv
    test_df = pd.read_csv(os.path.join(args.input_dir, 'submission_format.csv'))
    test_meta_df = pd.read_csv(os.path.join(args.input_dir, 'test_metadata.csv'))
    test_df = pd.merge(test_meta_df, test_df, on='filename', how='left')
    test_df['file'] = test_df['filename'].map(lambda x: os.path.join(args.input_dir, x))
    print('N examples:', len(test_df))

    # Clean expected text
    test_df['expected_text'] = test_df['expected_text'].fillna('empty text')
    test_df.loc[test_df['expected_text'] == '', 'expected_text'] = 'empty text'
    test_df['expected_text'] = test_df['expected_text'].map(clean)

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    # Init speech generation model
    processor_tts = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    model = model.to(device)
    vocoder = vocoder.to(device)
    speaker_embeddings = torch.tensor(np.load(os.path.join(args.assets_dir, 'Matthijs-cmu-arctic-xvectors-7900.npy')))

    #---- SPEECH GENERATION MODEL ---------------------------------------------

    # Load pre-generated audio for all "expected_text" from the train.
    # If test has a match we skip generation
    with open(os.path.join(args.assets_dir, 'gen_voice_map_fp16_seed33.pkl'), 'rb') as f:
        gen_voice_map = pickle.load(f)

    set_seed(33)

    for counter, (name, text) in enumerate(zip(test_df['filename'], test_df['expected_text'])):
        if text not in gen_voice_map:
            inputs = processor_tts(text=text, return_tensors='pt')
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                speech = model.generate(inputs['input_ids'].to(device), speaker_embeddings=speaker_embeddings.to(device), vocoder=vocoder)
            speech = speech.detach().cpu().numpy().astype(np.float32)
            gen_voice_map[text] = speech
        print('line:', counter, end='\r')

    #---- TRANSCRIPTION MODEL -------------------------------------------------

    print('Init datasets...')
    # We use only "whisper-medium.en" architecture for transcription due to time constraints
    # despite that we use different architectures for classification.
    # Using corresponding native transcription from each architecture used for classification
    # possibly may improve the score
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium.en')
    test_dataset = TranscriptionDatasetInfer(test_df, proc_image=processor)
    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False,
                        pin_memory=True,)

    print('Init model tr...')
    config = WhisperConfig.from_pretrained('openai/whisper-medium.en')
    model = WhisperForConditionalGeneration(config)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Compute maximum length for Whisper if not specified
    if args.max_length_whisper == -1 or args.max_length_whisper is None:
        max_length_whisper = get_max_length(test_df['expected_text'].unique(), processor.tokenizer)
    else:
        max_length_whisper = args.max_length_whisper

    # Using one fold for transcription
    dfs = []
    for model_id in range(1):
        weight = glob.glob(os.path.join(args.assets_dir, 'trans%d-*.bin' % model_id))[0]
        model.load_state_dict(torch.load(weight, map_location=device))
        print('Loaded model tr:', weight)

        preds_all_tr = []
        for batch_id, batch in enumerate(test_loader):
            x = batch['image'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                predicted_ids = model.generate(x, max_length=max_length_whisper)
                preds_tr = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                preds_all_tr.extend(preds_tr)
            print('Batch: %d of %d' % (batch_id, len(test_loader)), end='\r')

        test_df['trans'] = preds_all_tr
        # Clean transcription
        test_df['trans'] = test_df['trans'].fillna('empty transcription')
        test_df.loc[test_df['trans'] == '', 'trans'] = 'empty transcription'
        test_df['trans'] = test_df['trans'].map(clean)
        # Append copy i.e. each copy in the "dfs" list will have different "trans" col
        dfs.append(test_df.copy())

    # No need to repeaat becasue we use only single cls
    # dfs.append(dfs[0].copy())
    # dfs.append(dfs[1].copy())
    # dfs.append(dfs[0].copy())
    # dfs.append(dfs[1].copy())

    #---- CLASSIFICATION MODEL ------------------------------------------------

    # Compute maximum length for Deberta if not specified
    if args.max_length_deberta == -1 or args.max_length_deberta is None:
        concat_df = pd.concat(dfs)
        concat_list = concat_df['expected_text'] + ' ' + concat_df['trans']
        max_length_deberta = get_max_length(concat_list.unique(), tokenizer)
    else:
        max_length_deberta = args.max_length_deberta

    names = [
        'openai/whisper-medium.en',
        # 'openai/whisper-medium',
        # 'openai/whisper-small.en',
        # 'distil-whisper/distil-medium.en',
        # 'distil-whisper/distil-large-v3',
        # 'distil-whisper/distil-large-v2',
    ]
    preds_all = []
    for model_id, name in enumerate(names):
        print('Init datasets...')
        processor = WhisperProcessor.from_pretrained(name)
        # Each time we take "test_df" from the "dfs" list with different "trans" col
        test_dataset = ClassificationDataset(dfs[model_id], gen_voice_map, proc_image=processor, 
                                             proc_text=tokenizer, max_length=max_length_deberta)
        test_loader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            pin_memory=True,)

        print('Init model...')
        model = MultimodalClassifier(image_encoder_name=name, text_encoder_name='microsoft/deberta-v3-base')
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        weight = glob.glob(os.path.join(args.assets_dir, 'cls%d-*.bin' % model_id))[0]
        model.load_state_dict(torch.load(weight, map_location=device))
        print('Loaded model:', weight)

        preds = []
        for batch_id, batch in enumerate(test_loader):
            x = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                logits = model(x, input_ids, attention_mask)
                probas = torch.softmax(logits, axis=-1)
                preds.append(probas.detach().cpu().numpy()[:, 1:])
            print('Batch: %d of %d' % (batch_id, len(test_loader)), end='\r')

        preds_all.append(np.vstack(preds).ravel())

    test_df['score'] = np.mean(preds_all, axis=0)
    test_df[['filename', 'score']].to_csv(args.submission_file, index=False)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
