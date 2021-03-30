from __future__ import absolute_import, division, print_function

# ------------------------------------------------------------------------------ #
# base code from
#   https://github.com/huggingface/transformers/blob/master/examples/utils_ner.py
#   https://colab.research.google.com/github/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb
# ------------------------------------------------------------------------------ #

import os
import pdb

from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# Preprocessing
# ---------------------------------------------------------------------------- #

class InputExample(object):
    def __init__(self, guid, words, label):
        self.guid = guid
        self.words = words
        self.label = label

class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def read_examples_from_file(file_path, mode='train'):
    guid_index = 1
    examples = []
    tot_num_line = sum(1 for _ in open(file_path, 'r'))
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            sent, label = line.strip().split('\t')
            words = sent.split()
            assert(len(words) >= 1)
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                         words=words,
                                         label=label))
            guid_index += 1
    return examples

def convert_single_example_to_feature(example,
                                      label_map,
                                      max_seq_length,
                                      tokenizer,
                                      cls_token="[CLS]",
                                      cls_token_segment_id=0,
                                      sep_token="[SEP]",
                                      sep_token_extra=False,
                                      pad_token="[PAD]",
                                      pad_token_id=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0,
                                      ex_index=-1):

    tokens = []
    label = example.label
    label_id = -1
    for word in example.words:
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
    if label in label_map: label_id = label_map[label]
    if len(label.split()) >= 2: # logits as label
        label_id = label
    
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]

    # convention in BERT:
    # for single sequences:
    #  tokens:     [CLS] the dog is hairy . [SEP]
    #  input_ids:    x   x   x   x  x     x   x   0  0  0 ...
    #  segment_ids:  0   0   0   0  0     0   0   0  0  0 ...
    #  input_mask:   1   1   1   1  1     1   1   0  0  0 ...

    tokens += [sep_token]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token_id] * padding_length)
    input_mask += ([0] * padding_length)
    segment_ids += ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if ex_index != -1 and ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s", example.guid)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("label: %s", label)
        logger.info("label_id: %s", label_id)

    feature = InputFeature(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
    return feature

def convert_examples_to_features(examples,
                                 label_map,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_token="[PAD]",
                                 pad_token_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0):

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        feature = convert_single_example_to_feature(example,
                                                    label_map,
                                                    max_seq_length,
                                                    tokenizer,
                                                    cls_token=cls_token,
                                                    cls_token_segment_id=cls_token_segment_id,
                                                    sep_token=sep_token,
                                                    sep_token_extra=sep_token_extra,
                                                    pad_token=pad_token,
                                                    pad_token_id=pad_token_id,
                                                    pad_token_segment_id=pad_token_segment_id,
                                                    sequence_a_segment_id=sequence_a_segment_id,
                                                    ex_index=ex_index)
        features.append(feature)
    return features

# ---------------------------------------------------------------------------- #
# Long version
# ---------------------------------------------------------------------------- #

def create_long_model(model_type, model, tokenizer, config, attention_window=512, max_pos=4096):
    """Convert RoBERTa to Longformer.
    for other model_type like BERT, replacing model.encoder.layer.attention.self to LongformerSelfAttension()
    is not available at this time.
    """
    from transformers.modeling_longformer import LongformerSelfAttention
    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape

    if model_type in ['roberta']:
        max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2

    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos

    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    b = 0
    if model_type in ['roberta']: # NOTE: RoBERTa has positions 0,1 reserved
        k = 2
        step = current_max_pos - 2
        b = 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[b:]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    return model, tokenizer, config
