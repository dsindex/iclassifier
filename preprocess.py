from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import numpy as np
import random
import math
import json
from collections import Counter 
import pdb
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TRAIN_FILE = 'train.txt'
_VALID_FILE = 'valid.txt'
_TEST_FILE  = 'test.txt'
_SUFFIX = '.ids'
_VOCAB_FILE = 'vocab.txt'
_EMBED_FILE = 'embedding.npy'
_LABEL_FILE = 'label.txt'
_FSUFFIX = '.fs'

def build_label(input_path):
    logger.info("\n[building labels]")
    labels = {}
    label_id = 0
    tot_num_line = sum(1 for _ in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            sent, label = line.strip().split('\t')
            if label not in labels:
                labels[label] = label_id
                label_id += 1
    logger.info("\nUnique labels : {}".format(len(labels)))
    return labels

def write_label(labels, output_path):
    logger.info("\n[Writing label]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(labels.items())):
        label = item[0]
        label_id = item[1]
        f_write.write(label + ' ' + str(label_id))
        f_write.write('\n')
    f_write.close()

# ---------------------------------------------------------------------------- #
# Glove
# ---------------------------------------------------------------------------- #

def build_vocab_from_embedding(input_path, config):
    logger.info("\n[Building vocab from pretrained embedding]")
    vocab = {'<pad>':0, '<unk>':1}
    # build embedding as numpy array
    embedding = []
    # <pad>
    vector = np.array([float(0) for i in range(config['token_emb_dim'])]).astype(np.float)
    embedding.append(vector)
    # <unk>
    vector = np.array([random.random() for i in range(config['token_emb_dim'])]).astype(np.float)
    embedding.append(vector)
    tot_num_line = sum(1 for _ in open(input_path, 'r'))
    tid = len(vocab)
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            toks = line.strip().split()
            word = toks[0]
            vector = np.array(toks[1:]).astype(np.float)
            assert(config['token_emb_dim'] == len(vector))
            vocab[word] = tid
            embedding.append(vector)
            tid += 1
    embedding = np.array(embedding)
    return vocab, embedding
    
def build_data(input_path, tokenizer, vocab, config):
    logger.info("\n[Tokenizing and building data]")
    data = []
    all_tokens = Counter()
    _long_data = 0
    tot_num_line = sum(1 for _ in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            sent, label = line.strip().split('\t')
            tokens = tokenizer.tokenize(sent)
            if len(tokens) > config['n_ctx']:
                tokens = tokens[:config['n_ctx']]
                _long_data += 1
            for token in tokens:
                all_tokens[token] += 1
            data.append((tokens, label))
    logger.info("\n# Data over text length limit : {:,}".format(_long_data))
    logger.info("\nTotal unique tokens : {:,}".format(len(all_tokens)))
    logger.info("Vocab size : {:,}".format(len(vocab)))
    total_token_cnt = sum(all_tokens.values())
    cover_token_cnt = 0
    for item in all_tokens.most_common():
        if item[0] in vocab:
            cover_token_cnt += item[1]
    logger.info("Total tokens : {:,}".format(total_token_cnt))
    logger.info("Vocab coverage : {:.2f}%\n".format(cover_token_cnt/total_token_cnt*100.0))
    return data

def write_data(data, output_path, vocab, labels, config):
    logger.info("\n[Writing data]")
    unk, pad = vocab['<unk>'], vocab['<pad>']
    num_tok_per_sent = []
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(data)):
        sent, label = item[0], item[1]
        if len(sent) < 1: continue
        label_id = labels[label]
        f_write.write(str(label_id))
        for tok in sent:
            tok_id = unk if tok not in vocab else vocab[tok]
            f_write.write(' '+str(tok_id))
        num_tok_per_sent.append(len(sent))
        for _ in range(config['n_ctx'] - len(sent)):
            f_write.write(' '+str(pad))
        f_write.write('\n')
    f_write.close()
    ntps = np.array(num_tok_per_sent)
    logger.info("\nMEAN : {:.2f}, MAX:{}, MIN:{}, MEDIAN:{}\n".format(\
            np.mean(ntps), int(np.max(ntps)), int(np.min(ntps)), int(np.median(ntps))))

def write_vocab(vocab, output_path):
    logger.info("\n[Writing vocab]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(vocab.items())):
        tok = item[0]
        tok_id = item[1]
        f_write.write(tok + ' ' + str(tok_id))
        f_write.write('\n')
    f_write.close()

def write_embedding(embedding, output_path):
    logger.info("\n[Writing embedding]")
    np.save(output_path, embedding)

def preprocess_glove(config, options):
    from tokenizer import Tokenizer

    # vocab, embedding
    vocab, embedding = build_vocab_from_embedding(options.embedding_path, config)

    # build data
    tokenizer = Tokenizer(config)
    path = os.path.join(options.data_dir, _TRAIN_FILE)
    train_data = build_data(path, tokenizer, vocab, config)
    path = os.path.join(options.data_dir, _VALID_FILE)
    valid_data = build_data(path, tokenizer, vocab, config)
    path = os.path.join(options.data_dir, _TEST_FILE)
    test_data = build_data(path, tokenizer, vocab, config)

    # build labels
    path = os.path.join(options.data_dir, _TRAIN_FILE)
    labels = build_label(path)

    # write data, vocab, embedding, labels
    path = os.path.join(options.data_dir, _TRAIN_FILE + _SUFFIX)
    write_data(train_data, path, vocab, labels, config)
    path = os.path.join(options.data_dir, _VALID_FILE + _SUFFIX)
    write_data(valid_data, path, vocab, labels, config)
    path = os.path.join(options.data_dir, _TEST_FILE + _SUFFIX)
    write_data(test_data, path, vocab, labels, config)
    path = os.path.join(options.data_dir, _VOCAB_FILE)
    write_vocab(vocab, path)
    path = os.path.join(options.data_dir, _EMBED_FILE)
    write_embedding(embedding, path)
    path = os.path.join(options.data_dir, _LABEL_FILE)
    write_label(labels, path)

# ---------------------------------------------------------------------------- #
# BERT
#   reference
#     https://github.com/huggingface/transformers/blob/master/examples/run_ner.py
# ---------------------------------------------------------------------------- #

def build_features(input_path, tokenizer, labels, config, options):
    from util import read_examples_from_file
    from util import convert_examples_to_features

    logger.info("[Creating features from file] %s", input_path)
    examples = read_examples_from_file(input_path, mode='train')
    features = convert_examples_to_features(examples, labels, config['n_ctx'], tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            sequence_a_segment_id=0)
    return features

def write_features(features, output_path):
    import torch

    logger.info("[Saving features into file] %s", output_path)
    torch.save(features, output_path)
   
def preprocess_bert(config, options):
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(options.model_name_or_path,
                                              do_lower_case=options.do_lower_case)
    # build labels
    path = os.path.join(options.data_dir, _TRAIN_FILE)
    labels = build_label(path)

    # build features
    path = os.path.join(options.data_dir, _TRAIN_FILE)
    features = build_features(path, tokenizer, labels, config, options)
    path = os.path.join(options.data_dir, _VALID_FILE)
    features = build_features(path, tokenizer, labels, config, options)
    path = os.path.join(options.data_dir, _TEST_FILE)
    features = build_features(path, tokenizer, labels, config, options)

    # write features
    path = os.path.join(options.data_dir, _TRAIN_FILE + _FSUFFIX)
    write_features(features, path)
    path = os.path.join(options.data_dir, _VALID_FILE + _FSUFFIX)
    write_features(features, path)
    path = os.path.join(options.data_dir, _TEST_FILE + _FSUFFIX)
    write_features(features, path)

    # write labels
    path = os.path.join(options.data_dir, _LABEL_FILE)
    write_label(labels, path)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_path', type=str, default='embeddings/glove.6B.300d.txt')
    parser.add_argument('--config_path', type=str, default='config.json')
    parser.add_argument('--emb_class', type=str, default='glove', help='glove | bert')
    # for BERT
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    options = parser.parse_args()

    try:
        with open(options.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = dict()

    if options.emb_class == 'glove':
        preprocess_glove(config, options)
    if options.emb_class == 'bert' :
        preprocess_bert(config, options)


if __name__ == '__main__':
    main()
