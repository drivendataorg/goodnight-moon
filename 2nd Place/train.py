import os

MODEL_FILES = [
    'train_model3dsap.py',
    'train_model3mapd.py',
    'train_model3mb.py',
    'train_model3mcp.py',
    'train_model3scp.py',

    'train_model2dm.py',
    'train_model2ds.py',
    'train_model2m.py',
    'train_model2s.py',
    'train_model1m.py'
]
N_PARTS = 6
PARTS = [0, 1, 2, 3, 4, 5]

for model_file in MODEL_FILES:
    for p in PARTS:
        os.system(f'python {model_file} --fold {p} --n_folds {N_PARTS}')