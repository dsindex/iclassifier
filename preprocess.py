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
from util import load_config, Tokenizer
from datasets import Dataset
from transformers import AutoTokenizer

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
            toks = line.strip().split('\t')
            assert(len(toks) >= 2)
            label = toks[-1]
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
#   : single sentence classification
#   : sentence pair classification(TODO)
# ---------------------------------------------------------------------------- #

def build_init_vocab(config):
    init_vocab = {}
    init_vocab[config['pad_token']] = config['pad_token_id']
    init_vocab[config['unk_token']] = config['unk_token_id']
    return init_vocab

def build_vocab_from_embedding(input_path, vocab, config):
    """Build vocab from embedding file and init vocab(contains pad token and unk token only)
    """

    logger.info("\n[Building vocab from pretrained embedding]")
    # build embedding as numpy array
    embedding = []
    # <pad>
    vector = np.array([float(0) for i in range(config['token_emb_dim'])]).astype(float)
    embedding.append(vector)
    # <unk>
    vector = np.array([random.random() for i in range(config['token_emb_dim'])]).astype(float)
    embedding.append(vector)
    tot_num_line = sum(1 for _ in open(input_path, 'r'))
    tid = len(vocab)
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            toks = line.strip().split()
            word = toks[0]
            vector = np.array(toks[1:]).astype(float)
            assert(config['token_emb_dim'] == len(vector))
            vocab[word] = tid
            embedding.append(vector)
            tid += 1
    embedding = np.array(embedding)
    return vocab, embedding
    
def build_data(input_path, tokenizer):
    logger.info("\n[Tokenizing and building data]")
    vocab = tokenizer.vocab
    config = tokenizer.config
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

def write_data(data, output_path, tokenizer, labels):
    logger.info("\n[Writing data]")
    config = tokenizer.config
    pad_id = tokenizer.pad_id
    num_tok_per_sent = []
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(data)):
        tokens, label = item[0], item[1]
        if len(tokens) < 1: continue
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids_str = ' '.join([str(d) for d in ids])
        if len(label.split()) >= 2: # logits as label
            f_write.write(label + '\t' + ids_str)
        else:
            label_id = labels[label]
            f_write.write(str(label_id) + '\t' + ids_str)
        num_tok_per_sent.append(len(tokens))
        for _ in range(config['n_ctx'] - len(ids)):
            f_write.write(' '+str(pad_id))
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

def preprocess_glove(config):
    args = config['args']

    # vocab, embedding
    init_vocab = build_init_vocab(config)
    vocab, embedding = build_vocab_from_embedding(args.embedding_path, init_vocab, config)

    # build data
    tokenizer = Tokenizer(vocab, config)
    if args.augmented:
        path = os.path.join(args.data_dir, args.augmented_filename)
    else:
        path = os.path.join(args.data_dir, _TRAIN_FILE)
    train_data = build_data(path, tokenizer)

    path = os.path.join(args.data_dir, _VALID_FILE)
    valid_data = build_data(path, tokenizer)

    path = os.path.join(args.data_dir, _TEST_FILE)
    test_data = build_data(path, tokenizer)

    # build labels
    path = os.path.join(args.data_dir, _TRAIN_FILE)
    labels = build_label(path)

    # write data, vocab, embedding, labels
    if args.augmented:
        path = os.path.join(args.data_dir, args.augmented_filename + _SUFFIX)
    else:
        path = os.path.join(args.data_dir, _TRAIN_FILE + _SUFFIX)
    write_data(train_data, path, tokenizer, labels)

    path = os.path.join(args.data_dir, _VALID_FILE + _SUFFIX)
    write_data(valid_data, path, tokenizer, labels)

    path = os.path.join(args.data_dir, _TEST_FILE + _SUFFIX)
    write_data(test_data, path, tokenizer, labels)

    path = os.path.join(args.data_dir, _VOCAB_FILE)
    write_vocab(vocab, path)

    path = os.path.join(args.data_dir, _EMBED_FILE)
    write_embedding(embedding, path)

    path = os.path.join(args.data_dir, _LABEL_FILE)
    write_label(labels, path)

# ---------------------------------------------------------------------------- #
# BERT
#  : single sentence classification
#  : sentence pair classification
# ---------------------------------------------------------------------------- #

def build_dataset(input_path, labels):

    data = {'idx': [], 'label': [], 'sentence_a': [], 'sentence_b': []}
    tot_num_line = sum(1 for _ in open(input_path, 'r'))
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens) == 2:
                sentence_a = tokens[0]
                sentence_b = None
            if len(tokens) == 3:
                sentence_a = tokens[0]
                sentence_b = tokens[1]
            label = tokens[-1]
            if len(label.split()) == 1:
                if label == 'dummy': # see augment_data.py, --dummy_label
                    label_id = 0 # no matter what
                else:
                    label_id = labels[label]
            else:
                # soft label(logit), '-0.11 -0.89'
                label_id = label 
            data['idx'].append(idx)
            data['label'].append(label_id)
            data['sentence_a'].append(sentence_a)
            data['sentence_b'].append(sentence_b)
    logger.info("len(data['idx']): %s", len(data['idx']))
    dataset = Dataset.from_dict(data)
    logger.info("dataset desc: {}".format(dataset))
    logger.info("len(dataset): %s", len(dataset))
    return dataset

def build_encoded_dataset(input_path, tokenizer, labels, config, mode='train'):
    args = config['args']

    logger.info("[Creating encoded_dataset from file] %s", input_path)

    dataset = build_dataset(input_path, labels)

    def preprocess_function(examples):
        # https://huggingface.co/transformers/preprocessing.html#everything-you-always-wanted-to-know-about-padding-and-truncation
        if examples['sentence_b'][0] == None:
            return tokenizer(examples['sentence_a'], max_length=config['n_ctx'], padding='max_length', truncation=True)
        return tokenizer(examples['sentence_a'], examples['sentence_b'], max_length=config['n_ctx'], padding='max_length', truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # debugging
    need_token_type_ids = True
    if config['emb_class'] in ['roberta', 'bart', 'distilbert', 'ibert', 't5', 'gpt', 'gpt_neo', 'gptj']:
        need_token_type_ids = False
    logger.info("len(input_ids): %s", len(encoded_dataset['input_ids']))
    logger.info("len(attention_mask): %s", len(encoded_dataset['attention_mask']))
    if need_token_type_ids:
        logger.info("len(token_type_ids): %s", len(encoded_dataset['token_type_ids']))
    logger.info("len(label): %s", len(encoded_dataset['label']))
    logger.info("*** Example ***")
    for idx in range(10):
        logger.info("idx: %s", idx)
        input_ids = encoded_dataset['input_ids'][idx]
        attention_mask = encoded_dataset['attention_mask'][idx]
        if need_token_type_ids:
            token_type_ids = encoded_dataset['token_type_ids'][idx]
        label = encoded_dataset['label'][idx]
        logger.info("input_ids[idx]: %s", " ".join([str(x) for x in input_ids]))
        logger.info("decode(input_ids[idx]): %s", tokenizer.decode(input_ids))
        logger.info("attention_mask[idx]: %s", " ".join([str(x) for x in attention_mask]))
        if need_token_type_ids:
            logger.info("token_type_ids[idx]: %s", " ".join([str(x) for x in token_type_ids]))
        logger.info("label[idx]: %s", label)

    return encoded_dataset

def write_encoded_dataset(encoded_dataset, output_path):
    import torch

    logger.info("[Saving encoded_dataset into file] %s", output_path)
    torch.save(encoded_dataset, output_path)
   
def preprocess_bert(config):
    args = config['args']
    
    if config['emb_class'] == 'bart' and config['use_kobart']:
        from kobart import get_kobart_tokenizer
        tokenizer = get_kobart_tokenizer()
        tokenizer.cls_token = '<s>'
        tokenizer.sep_token = '</s>'
        tokenizer.pad_token = '<pad>'
    elif config['emb_class'] in ['gpt', 'gpt_neo', 'gptj']:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name_or_path, revision=args.bert_revision)
        if not tokenizer.pad_token:
            tokenizer.pad_token = '<pad>'
    elif config['emb_class'] in ['t5']:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name_or_path, revision=args.bert_revision)
        tokenizer.cls_token = '<s>'
        tokenizer.sep_token = '</s>'
        tokenizer.pad_token = '<pad>'
    elif config['emb_class'] in ['megatronbert']:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_name_or_path, revision=args.bert_revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name_or_path, revision=args.bert_revision)

    # build labels
    path = os.path.join(args.data_dir, _TRAIN_FILE)
    labels = build_label(path)

    # build encoded_dataset
    if args.augmented:
        path = os.path.join(args.data_dir, args.augmented_filename)
    else:
        path = os.path.join(args.data_dir, _TRAIN_FILE)
    train_encoded_dataset = build_encoded_dataset(path, tokenizer, labels, config, mode='train')

    path = os.path.join(args.data_dir, _VALID_FILE)
    valid_encoded_dataset = build_encoded_dataset(path, tokenizer, labels, config, mode='valid')

    path = os.path.join(args.data_dir, _TEST_FILE)
    test_encoded_dataset = build_encoded_dataset(path, tokenizer, labels, config, mode='test')

    # write encoded_dataset
    if args.augmented:
        path = os.path.join(args.data_dir, args.augmented_filename + _FSUFFIX)
    else:
        path = os.path.join(args.data_dir, _TRAIN_FILE + _FSUFFIX)
    write_encoded_dataset(train_encoded_dataset, path)

    path = os.path.join(args.data_dir, _VALID_FILE + _FSUFFIX)
    write_encoded_dataset(valid_encoded_dataset, path)

    path = os.path.join(args.data_dir, _TEST_FILE + _FSUFFIX)
    write_encoded_dataset(test_encoded_dataset, path)

    # write labels
    path = os.path.join(args.data_dir, _LABEL_FILE)
    write_label(labels, path)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove-cnn.json')
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_path', type=str, default='embeddings/glove.6B.300d.txt')
    parser.add_argument('--seed', default=42, type=int)
    # for Augmentation
    parser.add_argument('--augmented', action='store_true',
                        help="Set this flag to use augmented.txt for training or to use augmented.raw for labeling.")
    parser.add_argument('--augmented_filename', type=str, default='augmented.raw',
                        help="Filename for augmentation, augmented.raw or augmented.txt.")
    # for BERT
    parser.add_argument('--bert_model_name_or_path', type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased).")
    parser.add_argument('--bert_revision', type=str, default='main')
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # set config
    config = load_config(args)
    config['args'] = args
    logger.info("%s", config)

    if config['emb_class'] == 'glove':
        preprocess_glove(config)
    else:
        preprocess_bert(config)


if __name__ == '__main__':
    main()
