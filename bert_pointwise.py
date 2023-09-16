# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import csv
import os
os.environ["WANDB_MODE"]="offline"
import logging
import argparse
import random
import sys

from tqdm import tqdm, trange

from seqeval.metrics import f1_score as f1_score_seqeval, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertForMultilingualRep, BertForIntentClassificationAndSlotFilling, BertForTokenClassification
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel
# from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW, WarmupLinearSchedule
# from sentence_transformers import SentenceTransformer
from transformers.debug_utils import DebugUnderflowOverflow
from transformers.modeling_outputs import SequenceClassifierOutput

from configparser import ConfigParser
from pathlib import Path

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn
import wandb

# cfg = ConfigParser()
# thisfolder = os.path.dirname(os.path.abspath(__file__))
# cfg.read(os.path.join(thisfolder, 'path.ini'))
# addr = cfg.get('file_utils', 'path2')
PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               '.pytorch_pretrained_bert'))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class BertForSequenceClassificationBCE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        classifier_dropout = 0.5
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        weights=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).view(-1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # if self.config.problem_type == "regression":
            #     loss_fct = MSELoss()
            #     if self.num_labels == 1:
            #         loss = loss_fct(logits.squeeze(), labels.squeeze())
            #     else:
            #         loss = loss_fct(logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)
            # loss_fct = BCEWithLogitsLoss()
            loss_fct = BCEWithLogitsLoss(weight=weights)
            loss = loss_fct(logits, labels.float())
        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (logits,) + pooled_output
            return ((loss,) + output) if loss is not None else output, pooled_output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled_output,
            attentions=outputs.attentions,
        )




class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a = None, text_b=None, label=None, weight=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, weight=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.weight = weight


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, d):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        lines = []
        with open(input_file, "r", encoding='latin-1') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                if len(line) == 3:
                    if line[0] != line[1]:
                        lines.append(line)
            return lines


class MoocProcessor(DataProcessor):
    @classmethod
    def _read_mooc(cls, input_file, quotechar=None):
        lines = []
        with open(input_file, "r", encoding='latin-1') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                if len(line) == 5:
                    if line[0] == line[2]:
                        continue
                    tmp = []
                    tmp.append(line[0] + '\t' + line[1])
                    tmp.append(line[2] + '\t' + line[3])
                    tmp.append(line[4])
                    lines.append(tmp)
            return lines 

    def get_examples(self, data_dir, data_file, data_type, retrieval_augmentation=False):
        lines = self._read_mooc(os.path.join(data_dir, data_file))

        i = 0
        guid = "%s-%s" % (data_type, i)
        examples = []
        for line in lines:
            if retrieval_augmentation:
                a = line[0]
                b = line[1]
            # a = self.replace(line[0])
            # b = self.replace(line[1])
            # b = ''
            else:
                a = line[0].strip().split('\t')[0]
                b = line[1].strip().split('\t')[0]
            examples.append(InputExample(guid=guid, text_a=a, text_b=b, label=line[2]))
        return examples

    def replace(self, string):
        return string.replace('_', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ')

    def get_train_examples(self, data_dir, prefix, postfix='', retrieval_augmentation=False):
        # ml_train_1_e1_seed0_hard.txt
        # postfix = ''
        # return self.get_examples(data_dir, 'uc_train_1' + postfix + '.txt', 'train')
        return self.get_examples(data_dir, prefix + '_train_' + postfix + '.txt', 'train', retrieval_augmentation)
 
    def get_dev_examples(self, data_dir, prefix, postfix='', retrieval_augmentation=False):
        return self.get_examples(data_dir, prefix + '_test_' + postfix + '.txt', 'dev', retrieval_augmentation)

    def get_test_examples(self, data_dir, prefix, postfix='', retrieval_augmentation=False):
        return self.get_examples(data_dir, prefix + '_test_' + postfix + '.txt', 'test', retrieval_augmentation)

    def get_labels(self):
        return ['0', '1']


def convert_examples_to_features_seq(examples, label_list, max_seq_length, tokenizer):
    """With LSTM"""
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label: i for i, label in enumerate(label_list)}

    length = int(max_seq_length/2)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_a) > length - 2:
                tokens_a = tokens_a[:(length - 2)]
            if len(tokens_b) + len(tokens_b) > length - 2:
                tokens_b = tokens_b[:(length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # length = int(max_seq_length/2)

        if tokens_b:
            tokens_b = tokens_b + ["[SEP]"]
            tokens += tokens_b
            segment_ids += [1] * len(tokens_b)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad for token_a
        # length = max_seq_length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # if tokens_b:
        #     tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        #     tokens += tokens_b
        #     input_ids += tokenizer.convert_tokens_to_ids(tokens_b)
        #     segment_ids += [0] * len(tokens_b)  # for the second model
        #     input_mask += [1] * len(tokens_b)
        #     # pad for b
        #     padding = [0] * (max_seq_length - len(input_ids))
        #     input_ids += padding
        #     input_mask += padding
        #     segment_ids += padding

        # logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_id = float(example.label)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features


def convert_examples_to_features_weight(examples, label_list, max_seq_length, tokenizer):
    """With LSTM"""
    """Loads a data file into a list of `InputBatch`s."""

    i2w = {}
    dw = []
    with open('../bert/uc_c2w.txt', encoding='latin-1') as f:
        for line in f.readlines():
            # print(line)
            i, c, w = line.strip().split('\t')
            i2w[i] = (c, c + '\t' + w)

    with open('../NCF/w_uc.txt') as f:
        for line in f.readlines():
            c1, c2, l, check = line.strip().split('\t')
            if check == '0' and l == '1':
                dw.append((i2w[c1][1], i2w[c2][1]))
    
    print(len(dw))
    c = 0
    
    label_map = {label: i for i, label in enumerate(label_list)}

    length = int(max_seq_length/2)

    features = []
    for (ex_index, example) in enumerate(examples):
        # print(dw[0])
        # print(example.text_a)
        weight = 1
        if (example.text_a, example.text_b) in dw:
            weight = 1
            c += 1
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_a) > length - 2:
                tokens_a = tokens_a[:(length - 2)]
            if len(tokens_b) + len(tokens_b) > length - 2:
                tokens_b = tokens_b[:(length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # length = int(max_seq_length/2)

        if tokens_b:
            tokens_b = tokens_b + ["[SEP]"]
            tokens += tokens_b
            segment_ids += [1] * len(tokens_b)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad for token_a
        # length = max_seq_length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # if tokens_b:
        #     tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        #     tokens += tokens_b
        #     input_ids += tokenizer.convert_tokens_to_ids(tokens_b)
        #     segment_ids += [0] * len(tokens_b)  # for the second model
        #     input_mask += [1] * len(tokens_b)
        #     # pad for b
        #     padding = [0] * (max_seq_length - len(input_ids))
        #     input_ids += padding
        #     input_mask += padding
        #     segment_ids += padding

        # logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_id = float(example.label)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          weight=weight))
    print(c)

    return features


def convert_examples_to_features_pairwise(examples, label_list, max_seq_length, tokenizer):
    """With LSTM"""
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label: i for i, label in enumerate(label_list)}

    length = int(max_seq_length/2)

    pos, neg = [], []
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_a) > length - 2:
                tokens_a = tokens_a[:(length - 2)]
            if len(tokens_b) + len(tokens_b) > length - 2:
                tokens_b = tokens_b[:(length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # length = int(max_seq_length/2)

        if tokens_b:
            tokens_b = tokens_b + ["[SEP]"]
            tokens += tokens_b
            segment_ids += [1] * len(tokens_b)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad for token_a
        # length = max_seq_length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # if tokens_b:
        #     tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        #     tokens += tokens_b
        #     input_ids += tokenizer.convert_tokens_to_ids(tokens_b)
        #     segment_ids += [0] * len(tokens_b)  # for the second model
        #     input_mask += [1] * len(tokens_b)
        #     # pad for b
        #     padding = [0] * (max_seq_length - len(input_ids))
        #     input_ids += padding
        #     input_mask += padding
        #     segment_ids += padding

        # logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_id = float(example.label)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        if example.label == '1':
            pos.append(
                InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
        else:
            neg.append(
                InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    while len(neg) < len(pos):
        neg = neg + neg
    neg = neg[:len(pos)]
    assert len(pos) == len(neg)

    features = [pos, neg]
    print(len(pos))

    return features


def convert_examples_to_features_randompair(examples, label_list, max_seq_length, tokenizer):
    """With LSTM"""
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label: i for i, label in enumerate(label_list)}

    length = int(max_seq_length/2)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_a) > length - 2:
                tokens_a = tokens_a[:(length - 2)]
            if len(tokens_b) + len(tokens_b) > length - 2:
                tokens_b = tokens_b[:(length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # length = int(max_seq_length/2)

        if tokens_b:
            tokens_b = tokens_b + ["[SEP]"]
            tokens += tokens_b
            segment_ids += [1] * len(tokens_b)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad for token_a
        # length = max_seq_length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # if tokens_b:
        #     tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        #     tokens += tokens_b
        #     input_ids += tokenizer.convert_tokens_to_ids(tokens_b)
        #     segment_ids += [0] * len(tokens_b)  # for the second model
        #     input_mask += [1] * len(tokens_b)
        #     # pad for b
        #     padding = [0] * (max_seq_length - len(input_ids))
        #     input_ids += padding
        #     input_mask += padding
        #     segment_ids += padding

        # logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_id = float(example.label)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    num = len(features)
    pos, neg = [], []
    for i in range(4000):
        a, b = random.sample(features, 2)
        if float(a.label_id) > float(b.label_id):
            pos.append(a)
            neg.append(b)
        else:
            pos.append(b)
            neg.append(a)

    return [pos, neg]


def convert_examples_to_features_siamese(examples, label_list, max_seq_length, tokenizer):
    """With LSTM"""
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label: i for i, label in enumerate(label_list)}

    length = int(max_seq_length/2)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_a) > length - 2:
                tokens_a = tokens_a[:(length - 2)]
            if len(tokens_b) > length - 2:
                tokens_b = tokens_b[:(length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        length = int(max_seq_length/2)

        # if tokens_b:
        #     tokens_b = tokens_b + ["[SEP]"]
        #     tokens += tokens_b
        #     segment_ids += [1] * len(tokens_b)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad for token_a
        # length = max_seq_length
        padding = [0] * (length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        if tokens_b:
            tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
            tokens += tokens_b
            input_ids += tokenizer.convert_tokens_to_ids(tokens_b)
            segment_ids += [0] * len(tokens_b)  # for the second model
            input_mask += [1] * len(tokens_b)
            # pad for b
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

        # logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """With LSTM"""
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label: i for i, label in enumerate(label_list)}

    length = int(max_seq_length/2)

    features = []
    for (ex_index, example) in enumerate(examples):
        text_a1, text_a2 = example.text_a.split('\t')
        tokens_a1 = tokenizer.tokenize(text_a1)
        tokens_a2 = tokenizer.tokenize(text_a2)
        tokens_b = None
        if example.text_b:
            text_b1, text_b2 = example.text_b.split('\t')
            tokens_b1 = tokenizer.tokenize(text_b1)
            tokens_b2 = tokenizer.tokenize(text_b2)
            if len(tokens_a1) + len(tokens_a2) > length - 3:
                tokens_a2 = tokens_a2[:(length - len(tokens_a1) - 3)]
            if len(tokens_b1) + len(tokens_b2) > length - 3:
                tokens_b2 = tokens_b2[:(length - len(tokens_b1) - 3)]
            tokens_b = ["[CLS]"] + tokens_b1 + ["[SEP]"] + tokens_b2 + ["[SEP]"]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a1 + ["[SEP]"] + tokens_a2 + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a1) + 2) + [1] * (len(tokens_a2) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        length = int(max_seq_length/2)

        # if tokens_b:
        #     tokens_b = tokens_b + ["[SEP]"]
        #     tokens += tokens_b
        #     segment_ids += [1] * len(tokens_b)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad for token_a
        # length = max_seq_length
        padding = [0] * (length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        if tokens_b:
            # tokens_b = ["[CLS]"] + tokens_b1 + ["[SEP]"] + tokens_b2 + ["[SEP]"]
            segment_ids_b = [0] * (len(tokens_b1) + 2) + [1] * (len(tokens_b2) + 1)
            tokens += tokens_b
            input_ids += tokenizer.convert_tokens_to_ids(tokens_b)
            segment_ids += segment_ids_b  # for the second model
            input_mask += [1] * len(tokens_b)
            # pad for b
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

        # logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features


def accuracy(out, labels):
    if len(out.shape) == 2:
        outputs = np.argmax(out, axis=1)
        return np.sum(outputs == labels)
    if len(out.shape) == 3:
        outputs = np.argmax(out, axis=2).reshape(-1)
        return np.sum(outputs == labels.reshape(-1))

def accuracy_list(out, labels):
    a = 0
    for row_o, row_l in zip(out, labels):
        a += sum(1 for x, y in zip(row_o, row_l) if x == y) / float(len(row_o))
    a /= float(len(out))
    return a


def split_for_siamese(batch, length):
    input_ids, input_mask, segment_ids, label_ids = batch
    i1, i2 = input_ids[:, :length], input_ids[:, length:]
    a1, a2 = input_mask[:, :length], input_mask[:, length:]
    t1, t2 = segment_ids[:, :length], segment_ids[:, length:]
    input_ids = (i1, i2)
    input_mask = (a1, a2)
    segment_ids = (t1, t2)
    return input_ids, input_mask, segment_ids, label_ids

def split_for_pairwise(batch):
    input_ids, input_mask, segment_ids, label_ids, neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids = batch
    i = (input_ids, neg_input_ids)
    a = (input_mask, neg_input_mask)
    t = (segment_ids, neg_segment_ids)
    return i, a, t, label_ids


def train(model, optimizer, scheduler, train_examples, eval_examples, best_acc, processor, tokenizer, args):
    src_train_features = train_examples
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", args.num_train_steps)
    src_input_ids = torch.tensor([f.input_ids for f in src_train_features], dtype=torch.long)
    src_input_mask = torch.tensor([f.input_mask for f in src_train_features], dtype=torch.long)
    src_segment_ids = torch.tensor([f.segment_ids for f in src_train_features], dtype=torch.long)
    src_label_ids = torch.tensor([f.label_id for f in src_train_features], dtype=torch.long)
    src_weights = torch.tensor([f.weight for f in src_train_features], dtype=torch.long)
    train_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, src_weights)
    # train_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, src_input_ids, src_input_mask, src_segment_ids, src_label_ids)

    # src_train_features, neg_train_features = train_examples
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_examples[0]))
    # logger.info("  Batch size = %d", args.train_batch_size)
    # logger.info("  Num steps = %d", args.num_train_steps)
    # src_input_ids = torch.tensor([f.input_ids for f in src_train_features], dtype=torch.long)
    # src_input_mask = torch.tensor([f.input_mask for f in src_train_features], dtype=torch.long)
    # src_segment_ids = torch.tensor([f.segment_ids for f in src_train_features], dtype=torch.long)
    # src_label_ids = torch.tensor([f.label_id for f in src_train_features], dtype=torch.long)
    # neg_input_ids = torch.tensor([f.input_ids for f in neg_train_features], dtype=torch.long)
    # neg_input_mask = torch.tensor([f.input_mask for f in neg_train_features], dtype=torch.long)
    # neg_segment_ids = torch.tensor([f.segment_ids for f in neg_train_features], dtype=torch.long)
    # neg_label_ids = torch.tensor([f.label_id for f in neg_train_features], dtype=torch.long)
    # train_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # num_labels = 2
        # sid = epoch%20
        # label_list = processor.get_labels()
        # train_examples = processor.get_train_examples(args.data_dir, postfix='_e0_bce_' + str(sid))
        # src_train_features = convert_examples_to_features_weight(
        #     train_examples, label_list, args.max_seq_length, tokenizer)
        src_input_ids = torch.tensor([f.input_ids for f in src_train_features], dtype=torch.long)
        src_input_mask = torch.tensor([f.input_mask for f in src_train_features], dtype=torch.long)
        src_segment_ids = torch.tensor([f.segment_ids for f in src_train_features], dtype=torch.long)
        src_label_ids = torch.tensor([f.label_id for f in src_train_features], dtype=torch.long)
        src_weights = torch.tensor([f.weight for f in src_train_features], dtype=torch.long)
        train_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, src_weights)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = split_for_siamese(batch, int(args.max_seq_length/2))
            # input_ids, input_mask, segment_ids, label_ids = split_for_pairwise(batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, label_ids, weights = batch
            seq_output = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, weights=weights)
            loss = seq_output[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += label_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                args.global_step += 1
            sys.stdout.write(
                '\rMSE loss[{}] EnMSE loss[{}]'.format(loss.item(), 0))
            wandb.log({'train loss': loss.item()})

        # validation starts
        eval_s_features = eval_examples
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids)

        # eval_s_features = eval_examples
        # logger.info("***** Running evaluation *****")
        # logger.info("  Num examples = %d", len(eval_examples))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        # src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        # src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        # src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        # src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        # neg_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        # neg_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        # neg_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        # neg_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        # eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_l, true_l = [], []
        reps = []
        labels = []

        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            src_input_ids, src_input_mask, src_segment_ids, src_label_ids = batch
            # src_input_ids, src_input_mask, src_segment_ids, src_label_ids = split_for_siamese(batch, int(args.max_seq_length/2))
            # src_input_ids, src_input_mask, src_segment_ids, src_label_ids = split_for_pairwise(batch)
            

            with torch.no_grad():
                logits = model(src_input_ids, token_type_ids=src_segment_ids, attention_mask=src_input_mask)

            logits = logits[0].detach().cpu().numpy()
            label_ids = src_label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)
            outputs = np.int32(logits>0)
            tmp_eval_accuracy = np.sum(outputs == label_ids)/outputs.shape[0]

            labels.append(label_ids)
            # eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += src_label_ids.size(0)
            nb_eval_steps += 1
            # pred_labels = np.argmax(logits, axis=1)
            pred_labels = outputs

            print(pred_labels.shape, label_ids.shape)

            pred_l.append(pred_labels)
            true_l.append(label_ids)

       
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / (nb_eval_examples)
        pred_l = np.concatenate(pred_l, axis=None)
        true_l = np.concatenate(true_l, axis=None)
        f1_macro = f1_score(true_l, pred_l, average = 'macro')
        p_macro = precision_score(true_l, pred_l, average='macro')
        r_macro = recall_score(true_l, pred_l, average='macro')
        f1_micro = f1_score(true_l, pred_l, average = 'micro')
        p_micro = precision_score(true_l, pred_l, average='micro')
        r_micro = recall_score(true_l, pred_l, average='micro')

        # logger.info("pred: %s" % " ".join([str(x) for x in pred_l.tolist()]))
        # logger.info("true: %s" % " ".join([str(x) for x in true_l.tolist()]))

        logger.info("  acc = %f", eval_accuracy)
        logger.info("  macro f1 = %s", f1_macro)
        logger.info("  macro p = %s", p_macro)
        logger.info("  macro r = %s", r_macro)
        logger.info("  micro f1 = %s", f1_micro)
        logger.info("  micro p = %s", p_micro)
        logger.info("  micro r = %s", r_micro)
        logger.info("  loss = %f", eval_loss)
        
        if f1_macro > best_acc:
        #  if True:
            best_acc = f1_macro
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), args.output_model_file)
            es = 0
        else:
            es += 1
            print("Counter {} of 5".format(es))
        # if es > 4:
        #     print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", f1_macro, "...")
        #     break

        # validation starts
        # eval_s_features = src_train_features
        # logger.info("***** Running evaluation *****")
        # logger.info("  Num examples = %d", len(eval_examples))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        # src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        # src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        # src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        # src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        # eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids)

        # # eval_s_features = eval_examples
        # # logger.info("***** Running evaluation *****")
        # # logger.info("  Num examples = %d", len(eval_examples))
        # # logger.info("  Batch size = %d", args.eval_batch_size)
        # # src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        # # src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        # # src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        # # src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        # # neg_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
        # # neg_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
        # # neg_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
        # # neg_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
        # # eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids)

        # # Run prediction for full data
        # eval_sampler = SequentialSampler(eval_data)
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # model.eval()
        # eval_loss, eval_accuracy = 0, 0
        # nb_eval_steps, nb_eval_examples = 0, 0
        # pred_l, true_l = [], []
        # reps = []
        # labels = []

        # for batch in eval_dataloader:
        #     batch = tuple(t.to(args.device) for t in batch)
        #     src_input_ids, src_input_mask, src_segment_ids, src_label_ids = batch
        #     # src_input_ids, src_input_mask, src_segment_ids, src_label_ids = split_for_siamese(batch, int(args.max_seq_length/2))
        #     # src_input_ids, src_input_mask, src_segment_ids, src_label_ids = split_for_pairwise(batch)
            

        #     with torch.no_grad():
        #         logits = model(src_input_ids, token_type_ids=src_segment_ids, attention_mask=src_input_mask)

        #     logits = logits[0].detach().cpu().numpy()
        #     label_ids = src_label_ids.to('cpu').numpy()
        #     # tmp_eval_accuracy = accuracy(logits, label_ids)
        #     outputs = np.int32(logits>0)
        #     tmp_eval_accuracy = np.sum(outputs == label_ids)/outputs.shape[0]

        #     labels.append(label_ids)
        #     # eval_loss += tmp_eval_loss.mean().item()
        #     eval_accuracy += tmp_eval_accuracy

        #     nb_eval_examples += src_label_ids.size(0)
        #     nb_eval_steps += 1
        #     # pred_labels = np.argmax(logits, axis=1)
        #     pred_labels = outputs

        #     print(pred_labels.shape, label_ids.shape)

        #     pred_l.append(pred_labels)
        #     true_l.append(label_ids)

       
        # eval_loss = eval_loss / nb_eval_steps
        # eval_accuracy = eval_accuracy / (nb_eval_examples)
        # pred_l = np.concatenate(pred_l, axis=None)
        # true_l = np.concatenate(true_l, axis=None)
        # f1_macro = f1_score(true_l, pred_l, average = 'macro')
        # p_macro = precision_score(true_l, pred_l, average='macro')
        # r_macro = recall_score(true_l, pred_l, average='macro')
        # f1_micro = f1_score(true_l, pred_l, average = 'micro')
        # p_micro = precision_score(true_l, pred_l, average='micro')
        # r_micro = recall_score(true_l, pred_l, average='micro')

        # # logger.info("pred: %s" % " ".join([str(x) for x in pred_l.tolist()]))
        # # logger.info("true: %s" % " ".join([str(x) for x in true_l.tolist()]))

        # logger.info("  acc = %f", eval_accuracy)
        # logger.info("  macro f1 = %s", f1_macro)
        # logger.info("  macro p = %s", p_macro)
        # logger.info("  macro r = %s", r_macro)
        # logger.info("  micro f1 = %s", f1_micro)
        # logger.info("  micro p = %s", p_micro)
        # logger.info("  micro r = %s", r_micro)
        # logger.info("  loss = %f", eval_loss)
        
        model.train()

    return best_acc, model

def eval(model, eval_examples, args):    
    eval_s_features = eval_examples
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
    src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
    src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
    src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
    eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids)

    # eval_s_features = eval_examples
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_examples))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    # src_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
    # src_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
    # src_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
    # src_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
    # neg_input_ids = torch.tensor([f.input_ids for f in eval_s_features], dtype=torch.long)
    # neg_input_mask = torch.tensor([f.input_mask for f in eval_s_features], dtype=torch.long)
    # neg_segment_ids = torch.tensor([f.segment_ids for f in eval_s_features], dtype=torch.long)
    # neg_label_ids = torch.tensor([f.label_id for f in eval_s_features], dtype=torch.long)
    # eval_data = TensorDataset(src_input_ids, src_input_mask, src_segment_ids, src_label_ids, neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_l, true_l = [], []
    reps = []
    labels = []
    cons_r = 0
    emb = []

    
    #  for batch in eval_dataloader:
    for _, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t.to(args.device) for t in batch)
        # src_input_ids, src_input_mask, src_segment_ids, src_label_ids = split_for_siamese(batch, int(args.max_seq_length/2))
        src_input_ids, src_input_mask, src_segment_ids, src_label_ids = batch
        # src_input_ids, src_input_mask, src_segment_ids, src_label_ids = split_for_pairwise(batch)

        with torch.no_grad():
            # logits = model(src_input_ids, token_type_ids=src_segment_ids, attention_mask=src_input_mask, return_dict=True)
            logits = model(src_input_ids, token_type_ids=src_segment_ids, attention_mask=src_input_mask)


        logits_sigmoid = torch.sigmoid(logits[0])
        logits_sigmoid = logits_sigmoid.detach().cpu().numpy()
        pooled_output = logits[1].detach().cpu().numpy()
        print(pooled_output.shape)
        # np.save('uc_bert.npy', pooled_output)

        logits = logits[0]
        
        logits = logits.detach().cpu().numpy()
        # outputs = np.argmax(logits, axis=1)
        # print(logits.shape, outputs.sum())
        label_ids = src_label_ids.to('cpu').numpy()
        # tmp_eval_accuracy = accuracy(logits, label_ids)
        outputs = np.int32(logits>0)
        tmp_eval_accuracy = np.sum(outputs == label_ids)/outputs.shape[0]

        labels.append(label_ids)
        # eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += src_label_ids.size(0)
        nb_eval_steps += 1
        # pred_labels = np.argmax(logits, axis=1)
        pred_labels = outputs

        pred_l.append(pred_labels)
        true_l.append(label_ids)
        # emb.append(pooled_output)
    
    # eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / (nb_eval_examples)
    pred_l = np.concatenate(pred_l, axis=None)
    true_l = np.concatenate(true_l, axis=None)
    # emb = np.concatenate(emb, axis=0)
    # print(emb.shape)
    # np.save('uc.npy', emb)
    f1_macro = f1_score(true_l, pred_l, average = 'macro')
    p_macro = precision_score(true_l, pred_l, average='macro')
    r_macro = recall_score(true_l, pred_l, average='macro')
    f1_micro = f1_score(true_l, pred_l, average = 'micro')
    p_micro = precision_score(true_l, pred_l, average='micro')
    r_micro = recall_score(true_l, pred_l, average='micro')

    # logger.info("pred: %s" % " ".join([str(x) for x in pred_l.tolist()]))
    # logger.info("true: %s" % " ".join([str(x) for x in true_l.tolist()]))

    logger.info("  seed = %f", args.seed)

    logger.info("  acc = %f", eval_accuracy)
    logger.info("  macro f1 = %s", f1_macro)
    logger.info("  macro p = %s", p_macro)
    logger.info("  macro r = %s", r_macro)
    logger.info("  micro f1 = %s", f1_micro)
    logger.info("  micro p = %s", p_micro)
    logger.info("  micro r = %s", r_micro)
    logger.info("  loss = %f", eval_loss)
    
    result = {
              "macro f1": f1_macro,
    "macro p": p_macro,
    "macro r": r_macro,
    "micro f1": f1_micro,
    "micro p": p_micro,
    "micro r": r_micro}
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return logits_sigmoid, outputs


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--old_output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume training.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1024,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--training_data',
                        type=str, default=None,
                        help="Training data")
    parser.add_argument('--retrieval_augmentation',
                        default=False,
                        action='store_true')
    parser.add_argument('--data_prefix',
                            type=str, default=None,
                            help="dataset prefix")
    parser.add_argument('--data_postfix',
                            type=str, default=None,
                            help="dataset postfix")

    args = parser.parse_args()

    wandb.init(project="bert", config=args, settings=wandb.Settings(start_method='fork'))

    processors = {
        'mooc': MoocProcessor
    }

    num_labels_task = {
        'mooc': 2
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # args.output_dir = cfg.get('file_utils', 'path4') + args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        print("load model from directory ({})".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()



    args.self_train = True

    train_examples = None
    num_train_steps = None
    if args.do_train:
        # train_examples = processor.get_train_examples(args.data_dir, postfix='_' + str(args.training_data))
        train_examples = processor.get_train_examples(args.data_dir, args.data_prefix, args.data_postfix, args.retrieval_augmentation)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.resume:
        print('Resume Training')
        # args.old_output_dir = cfg.get('file_utils', 'path4') + args.old_output_dir
        saved_output_model_file = os.path.join(args.old_output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(saved_output_model_file)
        model = BertForSequenceClassificationBCE.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)

    args.output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    args.n_gpu = n_gpu
    args.device = device
    args.num_train_steps = num_train_steps

    eval_examples = processor.get_dev_examples(args.data_dir, args.data_prefix, args.data_postfix, args.retrieval_augmentation)

    if args.do_train:
        src_train_features = convert_examples_to_features_weight(
            train_examples, label_list, args.max_seq_length, tokenizer)
    else:
        src_train_features = None

    eval_features = convert_examples_to_features_seq(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    best_acc = 0

    if args.do_train:
        args.tr_loss = 0
        args.nb_tr_steps = 1
        args.global_step = 0
        model = BertForSequenceClassificationBCE.from_pretrained(args.bert_model, 
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
            args.local_rank), num_labels=num_labels)
        debug_overflow = DebugUnderflowOverflow(model)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]
        t_total = num_train_steps
        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=args.learning_rate,
                                    bias_correction=False,
                                    max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            warmup_steps = int(args.warmup_proportion * t_total)
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
            # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

        args.t_total = t_total

        wandb.watch(model)

        best_acc, model = train(model, optimizer, scheduler, src_train_features, eval_features, best_acc, processor, tokenizer, args)


    model = BertForSequenceClassificationBCE.from_pretrained(args.bert_model, 
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                args.local_rank), num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        args.tr_loss = 0
        args.nb_tr_steps = 1
        args.global_step = 0
        test_examples = processor.get_test_examples(args.data_dir, args.data_prefix, args.data_postfix, args.retrieval_augmentation)
        test_features = convert_examples_to_features_seq(test_examples, label_list, args.max_seq_length, tokenizer)
        logits, outputs = eval(model, test_features, args)

        # train_examples = processor.get_train_examples(args.data_dir, postfix='_e0_bce_19')
        # train_features = convert_examples_to_features_seq(train_examples, label_list, args.max_seq_length, tokenizer)
        # logits, outputs = eval(model, train_features, args)

        i2w = {}
        # concepts = []
        # with open('uc_i2c.txt') as f:
        #     for line in f.readlines():
        #         i, c = line.strip().split('\t')
        #         concepts.append(c)
        # test_examples = []
        # for i1 in range(len(concepts)):
        #     test_examples.append(InputExample(guid=0, text_a=concepts[i1], text_b='', label='0'))
        # test_features = convert_examples_to_features_seq(
        #         test_examples, label_list, args.max_seq_length, tokenizer)
        # logits, outputs = eval(model, test_features, args)

        # print(logits, outputs)

        with open('../bert/uc_c2w.txt', encoding='latin-1') as f:
            for line in f.readlines():
                # print(line)
                i, c, w = line.strip().split('\t')
                i2w[int(i)] = (c, c + '\t' + w)
                # i2w[int(i)] = (c, c)
        
        for i1 in range(len(i2w)):
            test_examples = []
            for i2 in range(len(i2w)):
                test_examples.append(InputExample(guid=0, text_a=i2w[i1][1], text_b=i2w[i2][1], label='0'))
            test_features = convert_examples_to_features_seq(
                test_examples, label_list, args.max_seq_length, tokenizer)
            logits, outputs = eval(model, test_features, args)
            with open (args.training_data, 'a') as fw:
                for i in range(outputs.shape[0]):
                    # fw.write(i2w[i1][0] + '\t' + i2w[i][0] + '\t' + str(logits[i][0]) + '\t' + str(logits[i][1]) + '\t' + str(outputs[i]) + '\n')
                    fw.write(i2w[i1][0] + '\t' + i2w[i][0] + '\t' + str(1-logits[i]) + '\t' + str(logits[i]) + '\t' + str(outputs[i]) + '\n')


if __name__ == "__main__":
    main()