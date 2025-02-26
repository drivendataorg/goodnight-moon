import pickle
import torch
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np

model_configs = [
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
        'dataset_type': 'fixsr'
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
        'dataset_type': 'fixsr'
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
        'dataset_type': 'fixsr'
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
        'dataset_type': 'fixsr'
    },

    {
        'speech_processor': 'assets/processors/whisper_small_processor', 
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
        'dataset_type': 'fixsr'
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
        'dataset_type': 'base'
    },

    {
        'speech_processor': 'assets/processors/whisper_medium_processor', 
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
        'dataset_type': 'base'
    },

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
        'dataset_type': 'base'
    },

    {
        'speech_processor': 'assets/processors/whisper_small_processor', 
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
        'dataset_type': 'base'
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
        'dataset_type': 'base'
    },
]

class MeanPooling():
    pass

class AttentionPooling():
    pass

class CustomModel():
    pass

pred_list = []
for mconfig in model_configs:
    _labels = []
    _filenames = []
    _preds = []
    for path in mconfig['paths']:
        model = torch.load(path)
        _labels.extend(model['score'])
        _filenames.extend(model['filename'])
        _preds.extend(model['pred'])
    data_df = pd.DataFrame({'filename': _filenames, 'pred': _preds, 'score': _labels})
    if len(pred_list) == 0:
        base_data_df = data_df
        preds = base_data_df['pred'].values
    else:
        preds = base_data_df[['filename']].merge(data_df, how='inner', on='filename')['pred'].values
    pred_list.append(preds)
labels = base_data_df['score'].values


def find_weights(pred_list, labels, times=3):
    weights = np.ones([len(pred_list)])
    best_weights = weights * 1.0
    best_score = 1.0
    for _ in range(times):
        for i in range(len(pred_list)):
            for w in range(20, 200, 2):
                weights[i] = w / 100
                preds = sum([pred*wei for pred, wei in zip(pred_list, weights)]) / sum(weights)
                score = log_loss(labels, preds)
                if best_score > score:
                    best_weights = weights * 1.0
                    best_score = score
            weights = best_weights * 1.0
    print('best_score: ', best_score)
    return best_weights

best_weights = find_weights(pred_list, labels)
print('best_weights:', best_weights)


for weight, mconfig in zip(best_weights, model_configs):
    mconfig['weight'] = float(weight)

print('final configs: ', model_configs)
with open('model_configs.pkl', 'wb') as f:
    pickle.dump(model_configs, f)