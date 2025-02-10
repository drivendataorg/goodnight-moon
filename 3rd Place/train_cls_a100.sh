#------------------------------------------------------------------------------
# openai/whisper-medium.en
#------------------------------------------------------------------------------

for run in {0..6}
do

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_0_mediumen \
--image_encoder_name=openai/whisper-medium.en \
--tuned_transcriber_dir=tuned_transcriber_0_mediumen \
--transcriptions=transcriptions_0_mediumen.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=16 \
--accum=2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# openai/whisper-medium
#------------------------------------------------------------------------------

for run in {0..6}
do

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_1_medium \
--image_encoder_name=openai/whisper-medium \
--tuned_transcriber_dir=tuned_transcriber_1_medium \
--transcriptions=transcriptions_1_medium.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=16 \
--accum=2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# openai/whisper-small.en
#------------------------------------------------------------------------------

for run in {0..6}
do

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_2_smallen \
--image_encoder_name=openai/whisper-small.en \
--tuned_transcriber_dir=tuned_transcriber_2_smallen \
--transcriptions=transcriptions_2_smallen.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=32 \
--accum=1 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# distil-whisper/distil-medium.en
#------------------------------------------------------------------------------

for run in {0..6}
do

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_3_distmediumen \
--image_encoder_name=distil-whisper/distil-medium.en \
--tuned_transcriber_dir=tuned_transcriber_3_distmediumen \
--transcriptions=transcriptions_3_distmediumen.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=16 \
--accum=2 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# distil-whisper/distil-large-v3
# Note. For this model we use transcriptions from "openai/whisper-medium.en" 
# i.e. "transcriptions_0_mediumen.csv" instead of "transcriptions_4_distlargev3.csv"
#------------------------------------------------------------------------------

for run in {0..6}
do

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_4_distlargev3 \
--image_encoder_name=distil-whisper/distil-large-v3 \
--tuned_transcriber_dir=tuned_transcriber_4_distlargev3 \
--transcriptions=transcriptions_0_mediumen.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=8 \
--accum=4 \
--device=cuda:0

done

#------------------------------------------------------------------------------
# distil-whisper/distil-large-v2
#------------------------------------------------------------------------------

for run in {0..6}
do

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_5_distlargev2 \
--image_encoder_name=distil-whisper/distil-large-v2 \
--tuned_transcriber_dir=tuned_transcriber_5_distlargev2 \
--transcriptions=transcriptions_5_distlargev2.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=8 \
--accum=4 \
--device=cuda:0

done

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



