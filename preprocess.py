import sys
import os
import argparse
import numpy as np
import random
import math
import json
from collections import Counter 
import pdb

from tqdm import tqdm
from tokenizer import Tokenizer

_TRAIN_FILE = 'train.txt'
_VALID_FILE = 'valid.txt'
_SUFFIX = '.ids'
_VOCAB_FILE = 'vocab.txt'
_EMBED_FILE = 'embedding.txt'

def build_vocab_from_embedding(input_path, config):
    print("\n[Building vocab from pretrained embedding]")
    vocab = {'<pad>':0, '<unk>':1}
    vocab_size = max(len(vocab), config['vocab_size']-len(vocab))
    embedding = []
    # <pad>
    vector = [float(0) for i in range(config['token_emb_dim'])]
    embedding.append(vector)
    # <unk>
    vector = [random.random() for i in range(config['token_emb_dim'])]
    embedding.append(vector)
    tot_num_line = sum(1 for line in open(input_path, 'r'))
    tid = len(vocab)
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            toks = line.split()
            word = toks[0]
            vector = toks[1:]
            assert(config['token_emb_dim'] == len(vector))
            vocab[word] = tid
            embedding.append(vector)
            tid += 1
    return vocab, embedding
    

def build_data(input_path, tokenizer, vocab, config):
    print("\n[Tokenizing and building data]")
    data = []
    all_tokens = Counter()
    labels = {}
    label_id = 0
    _long_data = 0
    tot_num_line = sum(1 for line in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            sent, label = line.split('\t')
            tokens = tokenizer.tokenize(sent)
            if len(tokens) > config['n_ctx']:
                tokens = tokens[:config['n_ctx']]
                _long_data += 1
            for token in tokens:
                all_tokens[token] += 1
            data.append((tokens, label))
            if label not in labels:
                labels[label] = label_id
                label_id += 1
    print("\n# Data over text length limit : {:,}".format(_long_data))
    print("\nTotal unique tokens : {:,}".format(len(all_tokens)))
    print("Vocab size : {:,}".format(len(vocab)))
    total_token_cnt = sum(all_tokens.values())
    cover_token_cnt = 0
    for item in all_tokens.most_common():
        if item[0] in vocab:
            cover_token_cnt += item[1]
    print("Total tokens : {:,}".format(total_token_cnt))
    print("Vocab coverage : {:.2f}%\n".format(cover_token_cnt/total_token_cnt*100.0))
    print("\nUnique labels : {}".format(len(labels)))
    return data, labels


def write_data(data, output_path, vocab, labels, config):
    print("\n[Writing data]")
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
    print("\nMEAN : {:.2f}, MAX:{}, MIN:{}, MEDIAN:{}\n".format(\
            np.mean(ntps), int(np.max(ntps)), int(np.min(ntps)), int(np.median(ntps))))

def write_vocab(vocab, output_path):
    print("\n[Writing vocab]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(vocab.items())):
        tok = item[0]
        tok_id = item[1]
        f_write.write(tok + ' ' + str(tok_id))
        f_write.write('\n')
    f_write.close()

def write_embedding(embedding, output_path):
    print("\n[Writing embedding]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for tok_id, vector in enumerate(embedding):
        f_write.write(str(tok_id))
        for v in vector:
            f_write.write(' '+str(v))
        f_write.write('\n')
    f_write.close()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_path', type=str, default='embeddings/glove.6B.300d.txt')
    parser.add_argument('--config_path', type=str, default='config.json')
    options = parser.parse_args()

    try:
        with open(options.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = dict()

    # vocab, embedding
    vocab, embedding = build_vocab_from_embedding(options.embedding_path, config)

    # read data, labels
    tokenizer = Tokenizer(config)
    path = os.path.join(options.data_dir, _TRAIN_FILE)
    train_data, labels = build_data(path, tokenizer, vocab, config)
    path = os.path.join(options.data_dir, _VALID_FILE)
    valid_data, _ = build_data(path, tokenizer, vocab, config)

    # write data, vocab, embedding, labels
    path = os.path.join(options.data_dir, _TRAIN_FILE + _SUFFIX)
    write_data(train_data, path, vocab, labels, config)
    path = os.path.join(options.data_dir, _VALID_FILE + _SUFFIX)
    write_data(valid_data, path, vocab, labels, config)
    path = os.path.join(options.data_dir, _VOCAB_FILE)
    write_vocab(vocab, path)
    path = os.path.join(options.data_dir, _EMBED_FILE)
    write_embedding(embedding, path)


if __name__ == '__main__':
    main()
