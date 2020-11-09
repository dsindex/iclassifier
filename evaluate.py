from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import time
import pdb
import logging

import torch
import torch.quantization
import numpy as np

from tqdm import tqdm
from model import TextGloveGNB, TextGloveCNN, TextGloveDensenetCNN, TextGloveDensenetDSA, TextBertCNN, TextBertCLS
from util import load_config, to_device, to_numpy
from dataset import prepare_dataset, GloveDataset, BertDataset
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_path(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        opt.data_path = os.path.join(opt.data_dir, 'test.txt.ids')
    else:
        if opt.augmented:
            opt.data_path = os.path.join(opt.data_dir, 'augmented.raw.fs')
        else:
            opt.data_path = os.path.join(opt.data_dir, 'test.txt.fs')
    opt.embedding_path = os.path.join(opt.data_dir, 'embedding.npy')
    opt.label_path = os.path.join(opt.data_dir, 'label.txt')
    if opt.augmented:
        opt.test_path = os.path.join(opt.data_dir, 'augmented.raw')
    else:
        opt.test_path = os.path.join(opt.data_dir, 'test.txt')
    opt.vocab_path = os.path.join(opt.data_dir, 'vocab.txt')

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
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'gnb':
            model = TextGloveGNB(config, opt.embedding_path, opt.label_path)
        if config['enc_class'] == 'cnn':
            model = TextGloveCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-cnn':
            model = TextGloveDensenetCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-dsa':
            model = TextGloveDensenetDSA(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
    else:
        if config['emb_class'] == 'funnel':
            from transformers import FunnelTokenizer, FunnelConfig, FunnelBaseModel
            bert_config = FunnelConfig.from_pretrained(opt.bert_output_dir)
            bert_tokenizer = FunnelTokenizer.from_pretrained(opt.bert_output_dir)
            # FunnelBaseModel has no 'from_config'
            bert_model = FunnelBaseModel.from_pretrained(opt.bert_output_dir)
        else:
            from transformers import AutoTokenizer, AutoConfig, AutoModel
            bert_config = AutoConfig.from_pretrained(opt.bert_output_dir)
            bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_output_dir)
            bert_model = AutoModel.from_config(bert_config)
        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path)
    model.load_state_dict(checkpoint)
    model = model.to(opt.device)
    logger.info("[Model loaded]")
    return model

def convert_onnx(config, torch_model, x):
    opt = config['opt']
    import torch.onnx

    if config['emb_class'] == 'glove':
        input_names  = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch', 1: 'sequence'},
                        'output': {0: 'batch', 1: 'sequence'}}
    else:
        input_names  = ['input_ids', 'input_mask', 'segment_ids']
        output_names = ['output']
        dynamic_axes = {'input_ids': {0: 'batch', 1: 'sequence'},
                        'input_mask': {0: 'batch', 1: 'sequence'},
                        'segment_ids': {0: 'batch', 1: 'sequence'},
                        'output': {0: 'batch'}}
        
    with torch.no_grad():
        torch.onnx.export(torch_model,                  # model being run
                          x,                            # model input (or a tuple for multiple inputs)
                          opt.onnx_path,                # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=opt.onnx_opset, # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          verbose=True,
                          input_names=input_names,      # the model's input names
                          output_names=output_names,    # the model's output names
                          dynamic_axes=dynamic_axes)    # variable length axes

# ------------------------------------------------------------------------------ #
# source code from https://github.com/huggingface/transformers/blob/master/src/transformers/convert_graph_to_onnx.py#L374
# ------------------------------------------------------------------------------ #
def quantize_onnx(onnx_path, quantized_onnx_path):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU.
    """
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize

    onnx_model = onnx.load(onnx_path)

    # Discussed with @yufenglee from ONNX runtime, this will be address in the next release of onnxruntime
    print(
        "As of onnxruntime 1.4.0, models larger than 2GB will fail to quantize due to protobuf constraint.\n"
        "This limitation will be removed in the next release of onnxruntime."
    )

    quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )

    # Save model
    onnx.save_model(quantized_model, quantized_onnx_path)

def check_onnx(config):
    opt = config['opt']
    import onnx
    onnx_model = onnx.load(opt.onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

def convert_tvm(config, torch_model, x):
    opt = config['opt']
    import onnx
    import tvm
    from tvm import relay

    onnx_model = onnx.load(opt.onnx_path)
    logger.info("[ONNX model loaded]")
    batch_size = x[0].shape[0]
    seq_len = x[0].shape[1]
    shape_dict = {'input_ids': (batch_size, seq_len),
                  'input_mask': (batch_size, seq_len),
                  'segment_ids': (batch_size, seq_len), }
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, opset=opt.onnx_opset)  
    logger.info("[Converting to TVM done]")

    with open(os.path.join(opt.tvm_dir, 'model.json'), 'w') as fo:
        fo.write(tvm.ir.save_json(model))
    with open(os.path.join(opt.tvm_dir, 'model.params'), 'wb') as fo:
        fo.write(relay.save_param_dict(params))

# ---------------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------------- #

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
                if opt.augmented:
                    # print logits as label
                    logits = ['%.6f' % p for p in pred]
                    f.write(sent + '\t' + ' '.join(logits) + '\n')
                else:
                    pred_id = np.argmax(pred)
                    pred_label = labels[pred_id]
                    f.write(sent + '\t' + label + '\t' + pred_label + '\n')
    except Exception as e:
        logger.warn(str(e))

def prepare_datasets(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        DatasetClass = GloveDataset
    else:
        DatasetClass = BertDataset
    test_loader = prepare_dataset(config, opt.data_path, DatasetClass, sampling=False, num_workers=1)
    return test_loader

def evaluate(opt):
    # set config
    config = load_config(opt)
    if opt.num_threads > 0: torch.set_num_threads(opt.num_threads)
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

    # convert to onnx
    if opt.convert_onnx:
        (x, y) = next(iter(test_loader))
        x = to_device(x, opt.device)
        y = to_device(y, opt.device)
        convert_onnx(config, model, x)
        check_onnx(config)
        logger.info("[ONNX model saved] :{}".format(opt.onnx_path))
        # quantize onnx
        if opt.quantize_onnx:
            quantize_onnx(opt.onnx_path, opt.quantized_onnx_path)
            logger.info("[Quantized ONNX model saved] : {}".format(opt.quantized_onnx_path))
        return

    # load onnx model for using onnxruntime
    if opt.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = opt.num_threads
        sess_options.intra_op_num_threads = opt.num_threads
        ort_session = ort.InferenceSession(opt.onnx_path, sess_options=sess_options)

    # convert to tvm format
    if opt.convert_tvm:
        (x, y) = next(iter(test_loader))
        x = to_device(x, opt.device)
        y = to_device(y, opt.device)
        convert_tvm(config, model, x)
        logger.info("[TVM model saved] : {}".format(opt.tvm_path))
        return

    # enable to use dynamic quantized model (pytorch>=1.3.0)
    if opt.enable_dqm and opt.device == 'cpu':
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(model)

    # evaluation
    preds = None
    ys    = None
    correct = 0
    n_batches = len(test_loader)
    total_examples = 0
    whole_st_time = time.time()
    first_time = time.time()
    first_examples = 0
    total_duration_time = 0.0
    with torch.no_grad():
        for i, (x,y) in enumerate(tqdm(test_loader, total=n_batches)):
            start_time = time.time()
            x = to_device(x, opt.device)
            y = to_device(y, opt.device)

            if opt.enable_ort:
                x = to_numpy(x)
                if config['emb_class'] == 'glove':
                    ort_inputs = {ort_session.get_inputs()[0].name: x}
                else:
                    if config['emb_class'] in ['distilbert', 'bart']:
                        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                      ort_session.get_inputs()[1].name: x[1]}
                    else:
                        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                      ort_session.get_inputs()[1].name: x[1],
                                      ort_session.get_inputs()[2].name: x[2]}
                logits = ort_session.run(None, ort_inputs)[0]
                logits = to_device(torch.tensor(logits), opt.device)
            else:
                logits = model(x)

            if preds is None:
                preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)
            predicted = logits.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = y.size(0)
            total_examples += cur_examples
            if i == 0: # first one may take longer time, so ignore in computing duration.
                first_time = float((time.time()-first_time)*1000)
                first_examples = cur_examples
            if opt.num_examples != 0 and total_examples >= opt.num_examples:
                logger.info("[Stop Evaluation] : up to the {} examples".format(total_examples))
                break
            duration_time = float((time.time()-start_time)*1000)
            if i != 0: total_duration_time += duration_time
            '''
            logger.info("[Elapsed Time] : {}ms".format(duration_time))
            '''
    # generate report
    labels = model.labels
    label_names = [v for k, v in sorted(labels.items(), key=lambda x: x[0])] 
    preds_ids = np.argmax(preds, axis=1)
    try:
        print(classification_report(ys, preds_ids, target_names=label_names, digits=4)) 
        print(labels)
        print(confusion_matrix(ys, preds_ids))
    except Exception as e:
        logger.warn(str(e))

    acc  = correct / total_examples
    whole_time = float((time.time()-whole_st_time)*1000)
    avg_time = (whole_time - first_time) / (total_examples - first_examples)
    # write predictions to file
    write_prediction(opt, preds, labels)
    logger.info("[Accuracy] : {:.4f}, {:5d}/{:5d}".format(acc, correct, total_examples))
    logger.info("[Elapsed Time] : {}ms, {}ms on average".format(whole_time, avg_time))
    logger.info("[Elapsed Time(total_duration_time, average)] : {}ms, {}ms".format(total_duration_time, total_duration_time/(total_examples-1)))

# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {}
        for idx, line in enumerate(f):
            tokens = line.split()
            word = tokens[0]
            word_id = int(tokens[1])
            vocab[word] = word_id
        return vocab

def prepare_tokenizer(config, model):
    from tokenizer import Tokenizer
    opt = config['opt']
    if config['emb_class'] == 'glove':
        vocab = load_vocab(opt.vocab_path)
        tokenizer = Tokenizer(vocab, config)
    else:
        tokenizer = model.bert_tokenizer
    return tokenizer

def encode_text(config, tokenizer, text):
    if config['emb_class'] == 'glove':
        tokens = text.split()
        # kernel size can't be greater than actual input size,
        # we should pad the sequence up to the maximum kernel size + 1.
        min_seq_size = 10
        ids = tokenizer.convert_tokens_to_ids(tokens, pad_sequence=False, min_seq_size=min_seq_size)
        x = torch.tensor([ids])
        # x : [batch_size, variable size]
        # batch size: 1
    else:
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        if config['emb_class'] in ['bart', 'distilbert']:
            x = [inputs['input_ids'], inputs['attention_mask']]
            # x[0], x[1] : [batch_size, variable size]
        else:
            x = [inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']]
            # x[0], x[1], x[2] : [batch_size, variable size]
        # batch size: 1
    return x

def inference(opt):
    # set config
    config = load_config(opt)
    if opt.num_threads > 0: torch.set_num_threads(opt.num_threads)
    config['opt'] = opt

    # set path: opt.embedding_path, opt.vocab_path, opt.label_path
    set_path(config)
 
    # load pytorch model checkpoint
    checkpoint = load_checkpoint(config)

    # prepare model and load parameters
    model = load_model(config, checkpoint)
    model.eval()

    # load onnx model for using onnxruntime
    if opt.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = opt.num_threads
        sess_options.intra_op_num_threads = opt.num_threads
        ort_session = ort.InferenceSession(opt.onnx_path, sess_options=sess_options)

    # enable to use dynamic quantized model (pytorch>=1.3.0)
    if opt.enable_dqm and opt.device == 'cpu':
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(model)

    # prepare tokenizer
    tokenizer = prepare_tokenizer(config, model)

    # prepare labels
    labels = model.labels

    # inference
    f_out = open(opt.test_path + '.inference', 'w', encoding='utf-8')
    total_examples = 0
    total_duration_time = 0.0
    with torch.no_grad(), open(opt.test_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            start_time = time.time()
            sent, label = line.strip().split('\t')
            x_raw = sent.split()
            y_raw = label
            text = ' '.join(x_raw)
            x = encode_text(config, tokenizer, text)
            x = to_device(x, opt.device)

            if opt.enable_ort:
                x = to_numpy(x)
                if config['emb_class'] == 'glove':
                    ort_inputs = {ort_session.get_inputs()[0].name: x}
                else:
                    if config['emb_class'] in ['distilbert', 'bart']:
                        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                      ort_session.get_inputs()[1].name: x[1]}
                    else:
                        ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                                      ort_session.get_inputs()[1].name: x[1],
                                      ort_session.get_inputs()[2].name: x[2]}
                logits = ort_session.run(None, ort_inputs)[0]
                logits = to_device(torch.tensor(logits), opt.device)
            else:
                logits = model(x)

            predicted = logits.argmax(1)
            predicted = to_numpy(predicted)[0]
            predicted_raw = labels[predicted]
            f_out.write(text + '\t' + y_raw + '\t' + predicted_raw + '\n')
            total_examples += 1
            if opt.num_examples != 0 and total_examples >= opt.num_examples:
                logger.info("[Stop Inference] : up to the {} examples".format(total_examples))
                break
            duration_time = float((time.time()-start_time)*1000)
            if i != 0: total_duration_time += duration_time
            logger.info("[Elapsed Time] : {}ms".format(duration_time))
    f_out.close()
    logger.info("[Elapsed Time(total_duration_time, average)] : {}ms, {}ms".format(total_duration_time, total_duration_time/(total_examples-1)))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove-cnn.json')
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--model_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_examples', default=0, type=int, help="Number of examples to evaluate, 0 means all of them.")
    # for Augmentation
    parser.add_argument('--augmented', action='store_true',
                        help="Set this flag to generate augmented.raw.inference(augmented.txt) for training.")
    # for BERT
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    # for ONNX
    parser.add_argument('--convert_onnx', action='store_true',
                        help="Set this flag to convert to ONNX.")
    parser.add_argument('--enable_ort', action='store_true',
                        help="Set this flag to evaluate using ONNXRuntime.")
    parser.add_argument('--onnx_path', type=str, default='pytorch-model.onnx')
    parser.add_argument('--onnx_opset', default=11, type=int, help="ONNX opset version.")
    parser.add_argument('--quantize_onnx', action='store_true',
                        help="Set this flag to quantize ONNX.")
    parser.add_argument('--quantized_onnx_path', type=str, default='pytorch-model.onnx-quantized')
    # for TVM
    parser.add_argument('--convert_tvm', action='store_true',
                        help="Set this flag to convert ONNX to TVM.")
    parser.add_argument('--enable_tvm', action='store_true',
                        help="Set this flag to evaluate using TVM.(not implemented)")
    parser.add_argument('--tvm_dir', type=str, default='tvm-model')
    # for Quantization
    parser.add_argument('--enable_dqm', action='store_true',
                        help="Set this flag to use dynamic quantized model.")
    # for Inference
    parser.add_argument('--enable_inference', action='store_true',
                        help="Set this flag to inference for raw input text.")
    opt = parser.parse_args()

    if opt.enable_inference:
        inference(opt)
    else:
        evaluate(opt) 

if __name__ == '__main__':
    main()
