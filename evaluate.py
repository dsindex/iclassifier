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
        bert_config = Config.from_pretrained(opt.bert_output_dir)
        bert_tokenizer = Tokenizer.from_pretrained(opt.bert_output_dir,
                                                   do_lower_case=opt.bert_do_lower_case)
        # no need to use 'from_pretrained'
        bert_model = Model(bert_config)
        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    logger.info("[Model loaded]")
    return model

def convert_onnx(config, torch_model, x):
    opt = config['opt']
    import torch.onnx

    if config['emb_class'] == 'glove':
        input_names  = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
    if 'bert' in config['emb_class'] or 'bart' in config['emb_class']:
        input_names  = ['input_ids', 'input_mask', 'segment_ids']
        output_names = ['output']
        dynamic_axes = {'input_ids': {0: 'batch_size'},
                        'input_mask': {0: 'batch_size'},
                        'segment_ids': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
        
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      opt.onnx_path,             # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=True,
                      input_names=input_names,   # the model's input names
                      output_names=output_names, # the model's output names
                      dynamic_axes=dynamic_axes) # variable length axes

def check_onnx(config):
    opt = config['opt']
    import onnx
    onnx_model = onnx.load(opt.onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
 
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
    model.eval()

    # convert to onnx format
    if opt.convert_onnx:
        (x, y) = next(iter(test_loader))
        x = to_device(x, device)
        y = y.to(device)
        convert_onnx(config, model, x)
        check_onnx(config)
        logger.info("[ONNX model saved at {}".format(opt.onnx_path))
        return

    # load onnx model for using onnxruntime
    if opt.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = opt.num_threads
        sess_options.intra_op_num_threads = opt.num_threads
        ort_session = ort.InferenceSession(opt.onnx_path, sess_options=sess_options)

    # evaluation
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
            if opt.enable_ort:
                x = to_numpy(x)
                if config['emb_class'] == 'glove':
                    ort_inputs = {ort_session.get_inputs()[0].name: x}
                if 'bert' in config['emb_class'] or 'bart' in config['emb_class']:
                    ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                  ort_session.get_inputs()[1].name: x[1],
                                  ort_session.get_inputs()[2].name: x[2]}
                logits = ort_session.run(None, ort_inputs)[0]
                logits = to_device(torch.tensor(logits), device)
            else:
                logits = model(x)
            if preds is None:
                preds = to_numpy(logits)
            else:
                preds = np.append(preds, to_numpy(logits), axis=0)
            predicted = logits.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = y.size(0)
            total_examples += cur_examples
            if i == 0: # first one may take longer time, so ignore in computing duration.
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
    # for ONNX
    parser.add_argument('--convert_onnx', action='store_true',
                        help="Set this flag to convert to onnx format.")
    parser.add_argument('--enable_ort', action='store_true',
                        help="Set this flag to evaluate using onnxruntime.")
    parser.add_argument('--onnx_path', type=str, default='pytorch-model.onnx')
    opt = parser.parse_args()

    evaluate(opt) 

if __name__ == '__main__':
    main()
