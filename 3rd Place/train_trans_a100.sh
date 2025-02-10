#------------------------------------------------------------------------------
# openai/whisper-medium.en
#------------------------------------------------------------------------------

for bunch in "0 1" "1 2" "2 3" "3 4" "4 5"
do 
set $bunch

python train_trans.py \
--input_dir=train \
--output_dir=tuned_transcriber_0_mediumen \
--image_encoder_name=openai/whisper-medium.en \
--batch_size=12 \
--accum=3 \
--initial_fold=$1 \
--final_fold=$2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# openai/whisper-medium
#------------------------------------------------------------------------------

for bunch in "0 1" "1 2" "2 3" "3 4" "4 5"
do 
set $bunch

python train_trans.py \
--input_dir=train \
--output_dir=tuned_transcriber_1_medium \
--image_encoder_name=openai/whisper-medium \
--batch_size=12 \
--accum=3 \
--initial_fold=$1 \
--final_fold=$2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# openai/whisper-small.en
#------------------------------------------------------------------------------

for bunch in "0 1" "1 2" "2 3" "3 4" "4 5"
do 
set $bunch

python train_trans.py \
--input_dir=train \
--output_dir=tuned_transcriber_2_smallen \
--image_encoder_name=openai/whisper-small.en \
--batch_size=18 \
--accum=2 \
--initial_fold=$1 \
--final_fold=$2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# distil-whisper/distil-medium.en
#------------------------------------------------------------------------------

for bunch in "0 1" "1 2" "2 3" "3 4" "4 5"
do 
set $bunch

python train_trans.py \
--input_dir=train \
--output_dir=tuned_transcriber_3_distmediumen \
--image_encoder_name=distil-whisper/distil-medium.en \
--batch_size=18 \
--accum=2 \
--initial_fold=$1 \
--final_fold=$2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# distil-whisper/distil-large-v3
#------------------------------------------------------------------------------

for bunch in "0 1" "1 2" "2 3" "3 4" "4 5"
do 
set $bunch

python train_trans.py \
--input_dir=train \
--output_dir=tuned_transcriber_4_distlargev3 \
--image_encoder_name=distil-whisper/distil-large-v3 \
--batch_size=9 \
--accum=4 \
--initial_fold=$1 \
--final_fold=$2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# distil-whisper/distil-large-v2
#------------------------------------------------------------------------------

for bunch in "0 1" "1 2" "2 3" "3 4" "4 5"
do 
set $bunch

python train_trans.py \
--input_dir=train \
--output_dir=tuned_transcriber_5_distlargev2 \
--image_encoder_name=distil-whisper/distil-large-v2 \
--batch_size=9 \
--accum=4 \
--initial_fold=$1 \
--final_fold=$2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

