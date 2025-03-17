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

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan, set_seed
from datasets import load_dataset

from utils import clean

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input_dir', default='train', type=str, help='Directory with dataset')
parser.add_argument('--output_dir', default='assets_new', type=str, help='Output directory')
parser.add_argument('--use_amp', default=1, type=int, choices=[0, 1], help='Whether to use auto mixed precision')
parser.add_argument('--device', default='cuda:0', type=str, help='Device')
args = parser.parse_args()
for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

device = torch.device(args.device)
os.makedirs(args.output_dir, exist_ok=True)

# Init text-to-speech models
processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
model = model.to(device)
vocoder = vocoder.to(device)

# Load and save speaker embeddings from a 'cmu-arctic-xvectors' dataset.
# Index 7900 was manually selected as female voice sounding similar to original audio
embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
speaker_embeddings = np.expand_dims(np.array(embeddings_dataset[7900]['xvector'], dtype=np.float32), axis=0)
np.save(os.path.join(args.output_dir, 'Matthijs-cmu-arctic-xvectors-7900.npy'), speaker_embeddings)
speaker_embeddings = torch.tensor(speaker_embeddings).to(device)

train_df = pd.read_csv(os.path.join(args.input_dir, 'train_metadata.csv'))

# Clean
train_df['expected_text'] = train_df['expected_text'].fillna('empty text')
train_df.loc[train_df['expected_text'] == '', 'expected_text'] = 'empty text'
train_df['expected_text'] = train_df['expected_text'].map(clean)

print('N unique:', train_df['expected_text'].nunique())

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Generate speech
set_seed(33)
gen_voice_map = {}
start = time.time()
for counter, text in enumerate(train_df['expected_text']):
    if text not in gen_voice_map:
        inputs = processor(text=text, return_tensors='pt')
        with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
            speech = model.generate(inputs['input_ids'].to(device), 
                                    speaker_embeddings=speaker_embeddings, 
                                    vocoder=vocoder)
        gen_voice_map[text] = speech.detach().cpu().numpy().astype(np.float32)
    print('line: %d    time: %d' % (counter, (time.time() - start)), end='\r')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Save
with open(os.path.join(args.output_dir, 'gen_voice_map_fp16_seed33.pkl'), 'wb') as f:
    pickle.dump(gen_voice_map, f)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

