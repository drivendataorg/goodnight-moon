# this is the format enhenced version

import os
import argparse

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import random
from datasets import Dataset, Audio, Features, ClassLabel, Array2D, Value
import datasets
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import WhisperForAudioClassification, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig, WhisperModel
import math
from typing import Optional, Tuple, Union
from typing import Any, List, Dict
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch import nn
from transformers import set_seed
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import json
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask
import numpy as np

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    TimeMask(
        min_band_part=0.1,
        max_band_part=0.15,
        fade=True,
        p=0.5,
    )
])

def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot_encode(labels, num_classes):
    # 初始化一个全零的数组
    one_hot = np.zeros((len(labels), num_classes))
    # 将对应的标签位置设置为1
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def softmax(x, axis=None):
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    logloss = log_loss(one_hot_encode(labels, 2), softmax(pred.predictions, axis=-1))

    return {"accuracy": acc, "f1": f1, 'logloss': logloss}


def load_to_dataset(audio_file_paths, tokens, decoder_mask, target):
    data = {'audio': audio_file_paths, 'decoder_input_ids': tokens,
            'decoder_attention_mask': decoder_mask, 'label': target}
    dataset = Dataset.from_dict(data)

    features = Features({
        'audio': Audio(sampling_rate=16000),
        'decoder_input_ids': datasets.Sequence(datasets.Value("int32")),  # ClassLabel(names=['control', 'mci', 'adrd'])
        'label': datasets.Sequence(datasets.Value("int32")),
        'decoder_attention_mask': datasets.Sequence(datasets.Value("int32")),
    })
    dataset = dataset.cast(features)

    return dataset


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = audio["array"]#feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0].astype(np.float16)
    del batch["audio"]
    # there is no need for encoding label
    batch["label"] = batch["label"]
    return batch


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class DataCollatorWithAugmentation:
    feature_extractor: Any
    augment_fn: Any  # Your augmentation function or pipeline
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract waveforms
        waveforms = [f["input_features"] for f in features]
        labels = [f["label"] for f in features]
        decoder_input_ids = [f["decoder_input_ids"] for f in features]
        decoder_attention_mask = [f["decoder_attention_mask"] for f in features]
        is_training_batch = all(f.get("is_train", False) for f in features)
        # Apply augmentation per sample
        if is_training_batch:
            waveforms = [self.augment_fn(np.array(w), sample_rate=16000) for w in waveforms]

        # Use the feature extractor to convert back to model inputs
        batch = self.feature_extractor(
            waveforms,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors=self.return_tensors,
            # padding="longest"
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["decoder_input_ids"] = torch.tensor(decoder_input_ids, dtype=torch.long)
        batch["decoder_attention_mask"] = torch.tensor(decoder_attention_mask, dtype=torch.long)

        return batch


class WhisperForConditionalGenerationMask(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.drop = nn.Dropout(p=0.0)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(self.drop(outputs[0]))

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            lm_logits = torch.gather(lm_logits, dim=2, index=decoder_input_ids.unsqueeze(-1)).squeeze(-1)[:, 4:]
            rpt = lm_logits.shape[1]
            # labels = labels.repeat(1, rpt)
            mask = decoder_attention_mask[:, 4:]
            lm_logits = lm_logits.masked_fill(mask == 0, float('nan'))
            lm_logits = torch.nanmean(lm_logits, dim=1)
            loss = loss_fct(lm_logits, labels.float().reshape(-1)).mean()
            # loss = loss_fct(lm_logits, labels.float())
            # loss = loss.masked_fill(mask == 0, float('nan'))
            # loss = torch.nanmean(loss, dim=1).mean()

            # loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            # loss = loss_fct(lm_logits, labels.float())

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai/whisper-small')
    parser.add_argument('--model_ckpt', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--warm', type=float, default=0.0)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    with open('./config.json') as fp:
        js = json.load(fp)
    print(f'Train epochs: {js["epochs"]}')
    args.epochs = js['epochs']
    print(args)

    set_seed(args.seed)
    for run_fold in range(1):
        global feature_extractor
        model_name = args.model_name # default: 'openai/whisper-small'
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        df = pd.read_csv('../fold.csv', index_col=0)
        # df = df[df.task.isin(['sentence_repetition'])]
        # debug here
        if args.debug:
            print('DDDEEEBBBUUUGGG!')
            df = df.sample(2000)
        trans = []
        for txt in df.expected_text.values:
            transcription = (
                "<|startoftranscript|>"
                "<|en|>"
                "<|transcribe|>"
                "<|notimestamps|>"
                f' {txt}'
            )
            trans.append(transcription)

        if model_name in ['openai/whisper-medium', 'openai/whisper-base']:
            processor = WhisperProcessor.from_pretrained('openai/whisper-small')
        else:
            processor = WhisperProcessor.from_pretrained(model_name)

        tok = processor.tokenizer(
            trans,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )  # Shape: [1, Sequence_Length]

        audio_file_paths = ['../../inputs/train/' + x for x in df.filename]

        tokens = tok.input_ids
        masks = tok.attention_mask
        labels = np.array([[x] for x in df.score.values.astype(int).tolist()])

        fold = run_fold
        fold_size = len(audio_file_paths) // 5 + 1

        # load_to_dataset(audio_file_paths, tokens, decoder_mask, target)
        train_ds = load_to_dataset(
            audio_file_paths=audio_file_paths,
            tokens=tokens,
            decoder_mask=masks,
            target=labels
        )

        train_ds = train_ds.map(prepare_dataset, remove_columns="audio")
        train_ds = train_ds.with_format("np")

        if args.model_ckpt != '':
            model = WhisperForConditionalGenerationMask.from_pretrained(args.model_ckpt)
        else:
            model = WhisperForConditionalGenerationMask.from_pretrained(model_name)

        del model.config.__dict__["max_length"]
        del model.config.__dict__["suppress_tokens"]
        del model.config.__dict__["begin_suppress_tokens"]
        batch_size = args.batch_size #16

        data_collator = DataCollatorWithAugmentation()
        data_collator.feature_extractor = feature_extractor
        data_collator.augment_fn = augment

        training_args = TrainingArguments(output_dir=f"../weights/{args.model_name.split('whisper-')[1]}-aug/",
                                          evaluation_strategy="no",
                                          learning_rate=args.lr, #2e-5,
                                          per_device_train_batch_size=batch_size,
                                          gradient_accumulation_steps=args.accum,
                                          per_device_eval_batch_size=batch_size,
                                          num_train_epochs=args.epochs,
                                          warmup_ratio=args.warm,
                                          logging_steps=10,
                                          bf16=True,
                                          save_strategy="steps",
                                          save_steps=230,
                                          load_best_model_at_end=False,
                                          metric_for_best_model="eval_loss",
                                          dataloader_num_workers=8,
                                          dataloader_drop_last=True,
                                          do_eval=False,
                                          report_to='tensorboard',
                                          run_name='Whisper-fine-tuning',
                                          remove_unused_columns=False)

        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_ds,
                          data_collator=data_collator,
                          tokenizer=feature_extractor)

        trainer.train()