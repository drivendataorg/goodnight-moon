from pathlib import Path

import nltk
nltk.data.path.append('./assets')

from huggingface_hub import hf_hub_download
from melo.api import TTS
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

Path("tts_data").mkdir(exist_ok=True)

nltk.download('averaged_perceptron_tagger_eng')

device = f"cuda"
model = TTS(language='EN_NEWEST', device=device, 
            config_path='assets/tts_model/config.json', ckpt_path='assets/tts_model/checkpoint.pth')
speaker_ids = model.hps.data.spk2id
print(speaker_ids)
speak_id = speaker_ids['EN-Newest']

train_metadata = pd.read_csv('data/train_metadata.csv')
texts = np.array(train_metadata['expected_text'].unique())
np.random.seed(42)
paths = []
for i, expected_text in tqdm(enumerate(texts)):
    speed = 0.7
    save_path1 = f"tts_data/{i}.wav"
    model.tts_to_file(expected_text, speak_id, save_path1, speed=speed, quiet=True)
    paths.append(save_path1)
pd.DataFrame({'path':paths, 'text': texts}).to_csv('tts_data/data_paths.csv', index=False)