#------------------------------------------------------------------------------
# openai/whisper-medium.en
#------------------------------------------------------------------------------

python train_cls.py \
--input_dir=train \
--output_dir=tuned_cls_0_mediumen \
--image_encoder_name=openai/whisper-medium.en \
--tuned_transcriber_dir=tuned_transcriber_0_mediumen \
--transcriptions=transcriptions_0_mediumen.csv \
--gen_voice_map=assets_new/gen_voice_map_fp16_seed33.pkl \
--batch_size=8 \
--accum=4 \
--device=cuda:0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



