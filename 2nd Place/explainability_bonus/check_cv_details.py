import pickle
import torch
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

metadata = pd.read_csv('data/train_metadata.csv')

medium_model_configs = [
    {
        'speech_processor': 'assets/processors/whisper_medium_processor', 
        'paths': [
            'assets/model3mapd_0.pth',
            'assets/model3mapd_1.pth',
            'assets/model3mapd_2.pth',
            'assets/model3mapd_3.pth',
            'assets/model3mapd_4.pth',
            'assets/model3mapd_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'attpooldov3',
    },
    
    {
        'speech_processor': 'assets/processors/whisper_medium_processor', 
        'paths': [
            'assets/model3mcp_0.pth',
            'assets/model3mcp_1.pth',
            'assets/model3mcp_2.pth',
            'assets/model3mcp_3.pth',
            'assets/model3mcp_4.pth',
            'assets/model3mcp_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'combpoolsv2',
    },

    {
        'speech_processor': 'assets/processors/whisper_medium_processor', 
        'paths': [
            'assets/model3mb_0.pth',
            'assets/model3mb_1.pth',
            'assets/model3mb_2.pth',
            'assets/model3mb_3.pth',
            'assets/model3mb_4.pth',
            'assets/model3mb_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'base',
    },

    {
        'speech_processor': 'assets/processors/whisper_medium_processor', 
        'paths': [
            'assets/model2m_0.pth',
            'assets/model2m_1.pth',
            'assets/model2m_2.pth',
            'assets/model2m_3.pth',
            'assets/model2m_4.pth',
            'assets/model2m_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'base',
    },

    {
        'speech_processor': 'assets/processors/distil_medium.en_processor', 
        'paths': [
            'assets/model2dm_0.pth',
            'assets/model2dm_1.pth',
            'assets/model2dm_2.pth',
            'assets/model2dm_3.pth',
            'assets/model2dm_4.pth',
            'assets/model2dm_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'base2',
    },

        {
        'speech_processor': 'assets/processors/whisper_medium_processor', 
        'paths': [
            'assets/model1m_0.pth',
            'assets/model1m_1.pth',
            'assets/model1m_2.pth',
            'assets/model1m_3.pth',
            'assets/model1m_4.pth',
            'assets/model1m_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'base',
    },
]

small_model_configs = [ 
    {
        'speech_processor': 'assets/processors/whisper_small_processor', 
        'paths': [
            'assets/model2s_0.pth',
            'assets/model2s_1.pth',
            'assets/model2s_2.pth',
            'assets/model2s_3.pth',
            'assets/model2s_4.pth',
            'assets/model2s_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'base2',
    },

    {
        'speech_processor': 'assets/processors/distil_small.en_processor', 
        'paths': [
            'assets/model2ds_0.pth',
            'assets/model2ds_1.pth',
            'assets/model2ds_2.pth',
            'assets/model2ds_3.pth',
            'assets/model2ds_4.pth',
            'assets/model2ds_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'base',
    },

    {
        'speech_processor': 'assets/processors/whisper_small_processor', 
        'paths': [
            'assets/model3scp_0.pth',
            'assets/model3scp_1.pth',
            'assets/model3scp_2.pth',
            'assets/model3scp_3.pth',
            'assets/model3scp_4.pth',
            'assets/model3scp_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'combpoolsv2',
    },

    {
        'speech_processor': 'assets/processors/distil_small.en_processor', 
        'paths': [
            'assets/model3dsap_0.pth',
            'assets/model3dsap_1.pth',
            'assets/model3dsap_2.pth',
            'assets/model3dsap_3.pth',
            'assets/model3dsap_4.pth',
            'assets/model3dsap_5.pth',
        ],
        'max_speech_len': 16,
        'batch_size': 64,
        'model_type': 'attpool',
    }
]

class MeanPooling():
    pass

class AttentionPooling():
    pass

class CustomModel():
    pass

tasks = metadata['task'].unique()
grades = metadata['grade'].unique()

print('medium models:---------------------------------------------------------')
for mconfig in medium_model_configs:
    for path in mconfig['paths']:
        model = torch.load(path)
        labels = model['score']
        preds = model['pred']
        loss = log_loss(labels, preds)
        print(f'{path}  overall cv loss:{loss}, number of data:{len(preds)}')

        for task in tasks:
            filenames = metadata[metadata['task']==task]['filename'].values
            ids = np.isin(model['filename'], filenames)
            loss = log_loss(labels[ids], preds[ids])
            print(f'task {task} cv loss:{loss}')
        for grade in grades:
            filenames = metadata[metadata['grade']==grade]['filename'].values
            ids = np.isin(model['filename'], filenames)
            loss = log_loss(labels[ids], preds[ids])
            print(f'grade {grade} cv loss:{loss}')
        print('')
    print('')
print('')
print('')

print('small models:---------------------------------------------------------')
for mconfig in small_model_configs:
    for path in mconfig['paths']:
        model = torch.load(path)
        labels = model['score']
        preds = model['pred']
        loss = log_loss(labels, preds)
        print(f'{path}  overall cv loss:{loss}, number of data:{len(preds)}')

        for task in tasks:
            filenames = metadata[metadata['task']==task]['filename'].values
            ids = np.isin(model['filename'], filenames)
            loss = log_loss(labels[ids], preds[ids])
            print(f'task {task} cv loss:{loss}')
        for grade in grades:
            filenames = metadata[metadata['grade']==grade]['filename'].values
            ids = np.isin(model['filename'], filenames)
            loss = log_loss(labels[ids], preds[ids])
            print(f'grade {grade} cv loss:{loss}')
        print('')
    print('')