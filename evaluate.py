from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import time
import pdb
import logging

import torch
import numpy as np

from tqdm import tqdm
from model import TextGloveCNN, TextGloveDensenetCNN, TextGloveDensenetDSA, TextBertCNN, TextBertCLS
from util import load_config, to_device, to_numpy
from dataset import prepare_dataset, SnipsGloveDataset, SnipsBertDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_prediction(opt, preds, labels):
    # load test data
    tot_num_line = sum(1 for _ in open(opt.test_path, 'r')) 
    with open(opt.test_path, 'r', encoding='utf-8') as f:
        data = []
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            sent, label = line.split('\t')
            data.append((sent, label))
    # write prediction
    try:
        pred_path = opt.test_path + '.pred'
        with open(pred_path, 'w', encoding='utf-8') as f:
            for entry, pred in zip(data, preds):
                sent, label = entry
                pred_id = np.argmax(pred)
                pred_label = labels[pred_id]
                f.write(sent + '\t' + label + '\t' + pred_label + '\n')
    except Exception as e:
        logger.warn(str(e))

def evaluate(opt):
    # set config
    config = load_config(opt)
    device = torch.device(opt.device)
    config['device'] = opt.device
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    if config['emb_class'] == 'glove':
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.ids')
    if 'bert' in config['emb_class']:
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.fs')
    opt.embedding_path = os.path.join(opt.data_dir, 'embedding.npy')
    opt.label_path = os.path.join(opt.data_dir, 'label.txt')
    opt.test_path = os.path.join(opt.data_dir, 'test.txt')

    test_data_path = opt.data_path
    torch.set_num_threads(opt.num_thread)

    # prepare test dataset
    if config['emb_class'] == 'glove':
        test_loader = prepare_dataset(opt, test_data_path, SnipsGloveDataset, shuffle=False, num_workers=1)
    if 'bert' in config['emb_class']:
        test_loader = prepare_dataset(opt, test_data_path, SnipsBertDataset, shuffle=False, num_workers=1)
 
    # load pytorch model checkpoint
    logger.info("[Loading model...]")
    if opt.device == 'cpu':
        checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(opt.model_path)

    # prepare model and load parameters
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'cnn':
            model = TextGloveCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-cnn':
            model = TextGloveDensenetCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-dsa':
            model = TextGloveDensenetDSA(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
    if 'bert' in config['emb_class']:
        from transformers import BertTokenizer, BertConfig, BertModel
        from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
        MODEL_CLASSES = {
            "bert": (BertConfig, BertTokenizer, BertModel),
            "albert": (AlbertConfig, AlbertTokenizer, AlbertModel)
        }
        Config    = MODEL_CLASSES[config['emb_class']][0]
        Tokenizer = MODEL_CLASSES[config['emb_class']][1]
        Model     = MODEL_CLASSES[config['emb_class']][2]

        bert_tokenizer = Tokenizer.from_pretrained(opt.bert_output_dir,
                                                   do_lower_case=opt.bert_do_lower_case)
        bert_model = Model.from_pretrained(opt.bert_output_dir)
        bert_config = bert_model.config
        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, opt.label_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info("[Loaded]")
 
    # evaluation
    model.eval()
    preds = None
    correct = 0
    n_batches = len(test_loader)
    total_examples = 0
    whole_st_time = time.time()
    first_time = time.time()
    first_examples = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(tqdm(test_loader, total=n_batches)):
            if type(x) != list: # torch.tensor
                x = x.to(device)
            else:               # list of torch.tensor
                for i in range(len(x)):
                    x[i] = x[i].to(device)
            y = y.to(device)
            logits = model(x)
            if preds is None:
                preds = to_numpy(logits)
            else:
                preds = np.append(preds, to_numpy(logits), axis=0)
            predicted = logits.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = y.size(0)
            total_examples += cur_examples
            if i == 0: # first one may takes longer time, so ignore in computing duration.
                first_time = int((time.time()-first_time)*1000)
                first_examples = cur_examples
    acc  = correct / total_examples
    whole_time = int((time.time()-whole_st_time)*1000)
    avg_time = (whole_time - first_time) / (total_examples - first_examples)
    # write predictions to file
    labels = model.labels
    write_prediction(opt, preds, labels)
    logger.info("[Accuracy] : {:.4f}, {:5d}/{:5d}".format(acc, correct, total_examples))
    logger.info("[Elapsed Time] : {}ms, {}ms on average".format(whole_time, avg_time))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--config', type=str, default='config-glove-cnn.json')
    parser.add_argument('--model_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', default=5, type=int, help="dummy for BaseModel.")
    # for BERT
    parser.add_argument('--bert_do_lower_case', action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()
