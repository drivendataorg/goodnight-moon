# Team sheep's solution doce

## 1. Environment & files
* please run following to create new conda environment
```shell
conda create --name literacy python==3.10
conda activate literacy
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install scikit-learn pandas transformers datasets librosa soundfile accelerate tensorboard audiomentations
```
* please organize the train and test samples at `../inputs` directory, after that, we will
have the following files. 
```shell
project_root/
├── inputs (data folder)
│   ├── train
│       └── ***.wav
│   └── test 
│       ├── test_metadata.csv
│       └── ***.wav
└── literacy (code folder)
    ├── src
    ├── weights
    ├── fold.csv
    └── Readme.MD (tile file)
```

## 2. Hardware requirement
* CPU: AMD 3960x
* RAM: 128G
* GPU: RTX A6000 48G
please run this code at single card with more than 48G vram.

## 3. Train
* please run `train.sh`, 
* or train separately five models
```shell
cd src
python v1.2.py --lr 3e-5 --model_name openai/whisper-base.en --epochs 1
python v1.2.py --lr 3e-5 --model_name openai/whisper-base.en --epochs 2
python v1.2.py --lr 3e-5 --model_name openai/whisper-base.en --epochs 3
python find_find_best_epochs.py

python v1.2_full_train.py --lr 3e-5 --model_name openai/whisper-base.en
python v1.2_aug_full.py --model_name openai/whisper-large-v3 --warm 0.2 --batch_size 4 --accum 32
python v1.2_full_train.py --model_name openai/whisper-medium.en --warm 0.2 --batch_size 4 --accum 32
python v1.2_full_train.py --model_name openai/whisper-large --warm 0.2 --batch_size 4 --accum 32
python v1.2_full_train.py --lr 3e-5 --model_name openai/whisper-small.en --warm 0.1
python v1.2_full_train.py --model_name openai/whisper-large-v3 --warm 0.2 --batch_size 4 --accum 32
cd ..
```
The output model file will be stored at `./weights`


## 4. Predict
* predict run `predict.py`
