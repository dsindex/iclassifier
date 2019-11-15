import sys
import os
import argparse
import json
import time
import pdb

import torch
from model import TextCNN
from dataset import SnipsDataset
from torch.utils.data import DataLoader

def evaluate(opt):
    test_data_path = opt.data_path
    embedding_path = opt.embedding_path
    label_path = opt.label_path
    model_path = opt.model_path
    batch_size = opt.batch_size
    device = opt.device
    config_path = opt.config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        config = dict()
 
    # load pytorch model
    print("[Loading model...]")
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_path)
    model = TextCNN(config, embedding_path, label_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()
    print ("[Loaded]")
 
    # prepare test dataset
    test_dataset = SnipsDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
            shuffle=False, num_workers=1, sampler=None)
    print("[Test data loaded]")

    # setting
    torch.set_num_threads(opt.num_thread)

    correct = 0
    total_examples = 0
    whole_st_time = time.time()
    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            predicted = output.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = x.size(0)
            total_examples += cur_examples
    acc  = correct / total_examples
    whole_time = int((time.time()-whole_st_time)*1000)
    avg_time = whole_time / total_examples

    print("[Accuracy] : {}, {}/{}".format(acc, correct, total_examples))
    print("[Elapsed Time] : {}ms, {}ms on average".format(whole_time, avg_time))

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
    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()
