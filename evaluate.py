from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import time
import pdb
import logging

import torch
from model import TextGloveCNN, TextBertCNN, TextBertCLS
from util import load_config, to_device, to_numpy
from dataset import prepare_dataset, SnipsGloveDataset, SnipsBertDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(opt):
    test_data_path = opt.data_path
    batch_size = opt.batch_size
    device = opt.device

    config = load_config(opt)
    torch.set_num_threads(opt.num_thread)

    # prepare test dataset
    if opt.emb_class == 'glove':
        test_loader = prepare_dataset(opt, test_data_path, SnipsGloveDataset, shuffle=False, num_workers=1)
    if 'bert' in opt.emb_class:
        test_loader = prepare_dataset(opt, test_data_path, SnipsBertDataset, shuffle=False, num_workers=1)
 
    # load pytorch model checkpoint
    logger.info("[Loading model...]")
    if device == 'cpu':
        checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(opt.model_path)

    # prepare model and load parameters
    if opt.emb_class == 'glove':
        model = TextGloveCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
    if 'bert' in opt.emb_class:
        from transformers import BertTokenizer, BertConfig, BertModel
        from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
        MODEL_CLASSES = {
            "bert": (BertConfig, BertTokenizer, BertModel),
            "albert": (AlbertConfig, AlbertTokenizer, AlbertModel)
        }
        Config    = MODEL_CLASSES[opt.emb_class][0]
        Tokenizer = MODEL_CLASSES[opt.emb_class][1]
        Model     = MODEL_CLASSES[opt.emb_class][2]

        bert_tokenizer = Tokenizer.from_pretrained(opt.bert_output_dir,
                                                   do_lower_case=opt.bert_do_lower_case)
        bert_model = Model.from_pretrained(opt.bert_output_dir)
        bert_config = bert_model.config
        ModelClass = TextBertCNN
        if opt.bert_model_class == 'TextBertCLS': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, opt.label_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info("[Loaded]")
 
    # evaluation
    model.eval()
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
            if opt.print_predicted_label:
                for p in predicted.cpu().numpy():
                    predicted_label = model.labels[p]
                    sys.stdout.write(predicted_label + '\n')
    acc  = correct / total_examples
    whole_time = int((time.time()-whole_st_time)*1000)
    avg_time = whole_time / total_examples

    logger.info("[Accuracy] : {:.4f}, {:5d}/{:5d}".format(acc, correct, total_examples))
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
    parser.add_argument('--emb_class', type=str, default='glove', help='glove | bert | albert')
    parser.add_argument("--print_predicted_label", action="store_true", help="Print predicted label out.")
    # for BERT
    parser.add_argument("--bert_do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--bert_output_dir", type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--bert_model_class', type=str, default='TextBertCNN',
                        help="model class, TextBertCNN | TextBertCLS")
    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()
