from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import time
import pdb
import logging

import torch
from model import TextGloveCNN, TextBertCNN
from dataset import SnipsGloveDataset, SnipsBertDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(opt):
    try:
        with open(opt.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        config = dict()
    return config

def prepare_dataset(opt, filepath, DatasetClass, shuffle=False, num_workers=1):
    dataset = DatasetClass(filepath)
    sampler = None
    loader = DataLoader(dataset, batch_size=opt.batch_size, \
            shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    logger.info("[{} data loaded]".format(filepath))
    return loader

def evaluate(opt):
    test_data_path = opt.data_path
    embedding_path = opt.embedding_path
    label_path = opt.label_path
    model_path = opt.model_path
    batch_size = opt.batch_size
    device = opt.device

    config = load_config(opt)
    torch.set_num_threads(opt.num_thread)

    # prepare test dataset
    if opt.emb_class == 'glove':
        test_loader = prepare_dataset(opt, test_data_path, SnipsGloveDataset, shuffle=False, num_workers=1)
    if opt.emb_class == 'bert':
        test_loader = prepare_dataset(opt, test_data_path, SnipsBertDataset, shuffle=False, num_workers=1)
 
    # load pytorch model checkpoint
    logger.info("[Loading model...]")
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_path)

    # prepare model and load parameters
    if opt.emb_class == 'glove':
        model = TextGloveCNN(config, embedding_path, label_path, emb_non_trainable=True)
    if opt.emb_class == 'bert':
        from transformers import BertTokenizer, BertConfig, BertModel
        bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_output_dir,
                                                       do_lower_case=opt.bert_do_lower_case)
        bert_model = BertModel.from_pretrained(opt.bert_output_dir)
        bert_config = bert_model.config
        model = TextBertCNN(config, bert_config, bert_model, opt.label_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info("[Loaded]")
 
    # evaluation
    model = model.eval()
    correct = 0
    total_examples = 0
    whole_st_time = time.time()
    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader):
            if type(x) != list: # torch.tensor
                x = x.to(device)
            else:               # list of torch.tensor
                for i in range(len(x)):
                    x[i] = x[i].to(device)
            y = y.to(device)
            output = model(x)
            predicted = output.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = y.size(0)
            total_examples += cur_examples
    acc  = correct / total_examples
    whole_time = int((time.time()-whole_st_time)*1000)
    avg_time = whole_time / total_examples

    logger.info("[Accuracy] : {}, {}/{}".format(acc, correct, total_examples))
    logger.info("[Elapsed Time] : {}ms, {}ms on average".format(whole_time, avg_time))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='data/snips/test.txt.ids')
    parser.add_argument('--embedding_path', type=str, default='data/snips/embedding.npy')
    parser.add_argument('--label_path', type=str, default='data/snips/label.txt')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--model_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--emb_class', type=str, default='glove', help='glove | bert')
    # for BERT
    parser.add_argument("--bert_do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--bert_output_dir", type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()