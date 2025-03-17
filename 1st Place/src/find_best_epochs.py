import os
import json
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import WhisperForAudioClassification, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import set_seed
from datasets import Dataset, Audio, Features, ClassLabel, Array2D

import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import metrics

import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import datasets
from torch.nn import CrossEntropyLoss

import librosa
from typing import Optional, Tuple, Union
import torch
import math
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, roc_auc_score

import warnings

warnings.filterwarnings('ignore')


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
            'decoder_attention_mask':decoder_mask, 'label': target}
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
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

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


class WhisperForConditionalGenerationMask(WhisperForConditionalGeneration):
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
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            lm_logits = torch.gather(lm_logits, dim=2, index=decoder_input_ids.unsqueeze(-1)).squeeze(-1)[:, 4:]
            rpt = lm_logits.shape[1]
            labels = labels.repeat(1, rpt)
            mask = decoder_attention_mask[:, 4:]
            loss = loss_fct(lm_logits, labels.float())
            loss = loss.masked_fill(mask == 0, float('nan'))
            loss = torch.nanmean(loss, dim=1).mean()

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


def get_oof_score(path):
    predicted, truths = [], []
    oof_logits = []
    for fold in range(5):
        label2id, id2label = dict(), dict()
        for i, label in enumerate(['wrong', 'correct']):
            label2id[label] = str(i)
            id2label[str(i)] = label
        global feature_extractor
        model_checkpoint = 'openai/whisper-base.en'
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        df = pd.read_csv('../fold.csv', index_col=0)
        # df = df.sample(2000)
        # df = df[df.task != 'nonword_repetition']
        # df = df[df.task == 'sentence_repetition']
        # df = df[df.task.isin(['blending', 'deletion'])]
        trans = []
        for tsk, txt in zip(df.task.values, df.expected_text.values):
            if tsk == 'sentence_repetition':
                transcription = (
                    "<|startoftranscript|>"
                    "<|en|>"
                    "<|transcribe|>"
                    "<|notimestamps|>"
                    f'{txt.capitalize()}'
                )

            else:
                transcription = (
                    "<|startoftranscript|>"
                    "<|en|>"
                    "<|transcribe|>"
                    "<|notimestamps|>"
                    f' {txt}'
                )
            trans.append(transcription)
        processor = WhisperProcessor.from_pretrained(model_checkpoint)
        tok = processor.tokenizer(
            trans,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )  # Shape: [1, Sequence_Length]
        audio_file_paths = ['../../inputs/train/' + x for x in df.filename]  # sorted(glob.glob('../inputs/train/*.wav'))
        # feats = torch.load('../notebooks/whisper_bert/roberta-large2.pth').float().numpy()[df.index.values]#[:2000]
        tokens = tok.input_ids
        masks = tok.attention_mask
        labels = np.array([[x] for x in df.score.values.astype(int).tolist()])
        fold_size = len(audio_file_paths) // 5 + 1
        # )
        valid_ds = load_to_dataset(
            audio_file_paths=audio_file_paths[fold * fold_size:(fold + 1) * fold_size],
            tokens=tokens[fold * fold_size:(fold + 1) * fold_size],
            decoder_mask=masks[fold * fold_size:(fold + 1) * fold_size],
            target=labels[fold * fold_size:(fold + 1) * fold_size])
        # train_ds = train_ds.map(prepare_dataset, remove_columns="audio")
        valid_ds = valid_ds.map(prepare_dataset, remove_columns="audio")
        # print(len(train_ds), len(valid_ds))
        # train_ds = train_ds.with_format("np")
        valid_ds = valid_ds.with_format("np")

        results_path = glob.glob(f'{path}/fold_{fold}/checkpoint-*/')
        model_checkpoint = sorted(results_path,
                                  key=lambda x: int(x.split('-')[-1][:-1]))[-1]

        model = WhisperForConditionalGenerationMask.from_pretrained(model_checkpoint).cuda()
        # emb = nn.Linear(in_features=768, out_features=2, bias=True)
        # model.set_output_embeddings(emb)
        # model.config.vocab_size = 2
        num_labels = len(id2label)

        dl = DataLoader(valid_ds, num_workers=4, batch_size=16, shuffle=False)

        pred = []
        labels = []
        logits = []
        with torch.no_grad():
            for data in tqdm(dl):
                label = data['label']
                del data['label']
                data = {k: v.cuda() for k, v in data.items()}
                with torch.cuda.amp.autocast():
                    r = model(**data)
                lm_logits = torch.gather(r['logits'], dim=2, index=data['decoder_input_ids'].unsqueeze(-1)).squeeze(-1)[
                            :, 4:]
                rpt = lm_logits.shape[1]
                mask = data['decoder_attention_mask'][:, 4:]
                masked_logit = lm_logits.masked_fill(mask == 0, float('nan'))
                logits.append(masked_logit.cpu())
                agg_logit = torch.nanmean(masked_logit, dim=1)
                pred.append(torch.sigmoid(agg_logit).cpu().numpy())
                labels.append(label.cpu().reshape(-1))
                # break

        preds = np.concatenate(pred)
        labels = np.concatenate(labels)

        predicted.append(preds)
        truths.append(labels)
        oof_logits.append(torch.cat(logits))

    prob = np.concatenate(predicted)
    truth = np.concatenate(truths)
    logit = torch.cat(oof_logits)

    df['prob'] = prob

    return df


if __name__ == '__main__':
    results = []
    for epoch in range(1, 4):
        rdf = get_oof_score(f'../weights/oof-base.en-epoch_{epoch}/')
        score = log_loss(rdf.score, rdf.prob)
        results.append({
            'epoch': epoch,
            'score': score
        })

    epoch_df = pd.DataFrame(results)
    best = int(epoch_df.sort_values('score').iloc[0].epoch)
    print('test epochs is {}'.format(best))
    with open('config.json', 'w') as fp:
        json.dump({'epochs': best}, fp)