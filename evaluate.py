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

def set_path(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.ids')
    if 'bert' in config['emb_class'] or 'bart' in config['emb_class']:
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.fs')
    opt.embedding_path = os.path.join(opt.data_dir, 'embedding.npy')
    opt.label_path = os.path.join(opt.data_dir, 'label.txt')
    opt.test_path = os.path.join(opt.data_dir, 'test.txt')

def prepare_datasets(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        DatasetClass = SnipsGloveDataset
    if 'bert' in config['emb_class'] or 'bart' in config['emb_class']:
        DatasetClass = SnipsBertDataset
    test_loader = prepare_dataset(opt, opt.data_path, DatasetClass, shuffle=False, num_workers=1)
    return test_loader

def load_checkpoint(config):
    opt = config['opt']
    if opt.device == 'cpu':
        checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(opt.model_path)
    logger.info("[Loading checkpoint done]")
    return checkpoint

def load_model(config, checkpoint):
    opt = config['opt']
    device = config['device']
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'cnn':
            model = TextGloveCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-cnn':
            model = TextGloveDensenetCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-dsa':
            model = TextGloveDensenetDSA(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
    if 'bert' in config['emb_class'] or 'bart' in config['emb_class']:
        from transformers import BertTokenizer, BertConfig, BertModel
        from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
        from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
        from transformers import BartConfig, BartTokenizer, BartModel
        MODEL_CLASSES = {
            "bert": (BertConfig, BertTokenizer, BertModel),
            "albert": (AlbertConfig, AlbertTokenizer, AlbertModel),
            "roberta": (RobertaConfig, RobertaTokenizer, RobertaModel),
            "bart": (BartConfig, BartTokenizer, BartModel)
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
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info("[Model loaded]")
    return model

def evaluate(opt):
    # set config
    config = load_config(opt)
    device = torch.device(opt.device)
    if opt.num_threads > 0: torch.set_num_threads(opt.num_threads)
    config['device'] = opt.device
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

    # prepare test dataset
    test_loader = prepare_datasets(config)
 
    # load pytorch model checkpoint
    checkpoint = load_checkpoint(config)

    # prepare model and load parameters
    model = load_model(config, checkpoint)

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
            x = to_device(x, device)
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
            if opt.num_examples != 0 and total_examples >= opt.num_examples:
                logger.info("[Stop Evaluation] : up to the {} examples".format(total_examples))
                break
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
    
    parser.add_argument('--config', type=str, default='configs/config-glove-cnn.json')
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--model_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_examples', default=0, type=int, help="number of examples to evaluate, 0 means all of them.")
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
