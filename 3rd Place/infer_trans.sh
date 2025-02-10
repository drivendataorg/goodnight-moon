#------------------------------------------------------------------------------
# openai/whisper-medium.en
#------------------------------------------------------------------------------

python infer_trans.py \
--input_dir=train \
--model_dir=tuned_transcriber_0_mediumen \
--output_file=transcriptions_0_mediumen.csv \
--image_encoder_name=openai/whisper-medium.en \
--batch_size=64 \
--device=cuda:0

#------------------------------------------------------------------------------
# openai/whisper-medium
#------------------------------------------------------------------------------

python infer_trans.py \
--input_dir=train \
--model_dir=tuned_transcriber_1_medium \
--output_file=transcriptions_1_medium.csv \
--image_encoder_name=openai/whisper-medium \
--batch_size=64 \
--device=cuda:0

#------------------------------------------------------------------------------
# openai/whisper-small.en
#------------------------------------------------------------------------------

python infer_trans.py \
--input_dir=train \
--model_dir=tuned_transcriber_2_smallen \
--output_file=transcriptions_2_smallen.csv \
--image_encoder_name=openai/whisper-small.en \
--batch_size=64 \
--device=cuda:0

#------------------------------------------------------------------------------
# distil-whisper/distil-medium.en
#------------------------------------------------------------------------------

python infer_trans.py \
--input_dir=train \
--model_dir=tuned_transcriber_3_distmediumen \
--output_file=transcriptions_3_distmediumen.csv \
--image_encoder_name=distil-whisper/distil-medium.en \
--batch_size=64 \
--device=cuda:0

#------------------------------------------------------------------------------
# distil-whisper/distil-large-v3
# Note. Transcriptions from openai/whisper-medium.en were used 
# to train a classifier based on this architecture
#------------------------------------------------------------------------------

# python infer_trans.py \
# --input_dir=train \
# --model_dir=tuned_transcriber_4_distlargev3 \
# --output_file=transcriptions_4_distlargev3.csv \
# --image_encoder_name=distil-whisper/distil-large-v3 \
# --batch_size=32 \
# --device=cuda:0

#------------------------------------------------------------------------------
# distil-whisper/distil-large-v2
#------------------------------------------------------------------------------

python infer_trans.py \
--input_dir=train \
--model_dir=tuned_transcriber_5_distlargev2 \
--output_file=transcriptions_5_distlargev2.csv \
--image_encoder_name=distil-whisper/distil-large-v2 \
--batch_size=32 \
--device=cuda:0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

