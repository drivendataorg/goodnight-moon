#! /bin/bash
cd src
# first find the best epoch, default should be 2 epochs
python v1.2.py --lr 3e-5 --model_name openai/whisper-base.en --epochs 1
python v1.2.py --lr 3e-5 --model_name openai/whisper-base.en --epochs 2
python v1.2.py --lr 3e-5 --model_name openai/whisper-base.en --epochs 3
python find_best_epochs.py
# run the full train
python v1.2_full_train.py --lr 3e-5 --model_name openai/whisper-base.en
python v1.2_aug_full.py --model_name openai/whisper-large-v3 --warm 0.2 --batch_size 4 --accum 32
python v1.2_full_train.py --model_name openai/whisper-medium.en --warm 0.2 --batch_size 4 --accum 32
python v1.2_full_train.py --model_name openai/whisper-large --warm 0.2 --batch_size 4 --accum 32
python v1.2_full_train.py --lr 3e-5 --model_name openai/whisper-small.en --warm 0.1
python v1.2_full_train.py --model_name openai/whisper-large-v3 --warm 0.2 --batch_size 4 --accum 32

cd ..