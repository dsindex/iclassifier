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
from diffq import DiffQuantizer

from tqdm import tqdm
from model import TextGloveGNB, TextGloveCNN, TextGloveDensenetCNN, TextGloveDensenetDSA, TextBertCNN, TextBertCLS, TextBertDensenetCNN
from transformers import AutoTokenizer, AutoConfig, AutoModel
from util import load_checkpoint, load_config, load_label, to_device, to_numpy, Tokenizer
from dataset import prepare_dataset, GloveDataset, BertDataset
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_path(config):
    args = config['args']
    if config['emb_class'] == 'glove':
        args.data_path = os.path.join(args.data_dir, 'test.txt.ids')
    else:
        if args.augmented:
            args.data_path = os.path.join(args.data_dir, 'augmented.raw.fs')
        else:
            args.data_path = os.path.join(args.data_dir, 'test.txt.fs')
    args.embedding_path = os.path.join(args.data_dir, 'embedding.npy')
    args.label_path = os.path.join(args.data_dir, 'label.txt')
    if args.augmented:
        args.test_path = os.path.join(args.data_dir, 'augmented.raw')
    else:
        args.test_path = os.path.join(args.data_dir, 'test.txt')
    args.vocab_path = os.path.join(args.data_dir, 'vocab.txt')

def load_model(config, checkpoint):
    args = config['args']
    labels = load_label(args.label_path)
    label_size = len(labels)
    config['labels'] = labels
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'gnb':
            model = TextGloveGNB(config, args.embedding_path, label_size)
        if config['enc_class'] == 'cnn':
            model = TextGloveCNN(config, args.embedding_path, label_size, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-cnn':
            model = TextGloveDensenetCNN(config, args.embedding_path, label_size, emb_non_trainable=True)
        if config['enc_class'] == 'densenet-dsa':
            model = TextGloveDensenetDSA(config, args.embedding_path, label_size, emb_non_trainable=True)
    else:
        if config['emb_class'] == 'bart' and config['use_kobart']:
            from transformers import BartModel
            from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
            bert_tokenizer = get_kobart_tokenizer()
            bert_tokenizer.cls_token = '<s>'
            bert_tokenizer.sep_token = '</s>'
            bert_tokenizer.pad_token = '<pad>'
            bert_model = BartModel.from_pretrained(get_pytorch_kobart_model())
            bert_config = bert_model.config
        elif config['emb_class'] in ['gpt']:    
            bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_output_dir)
            bert_tokenizer.bos_token = '<|startoftext|>'
            bert_tokenizer.eos_token = '<|endoftext|>'
            bert_tokenizer.cls_token = '<|startoftext|>'
            bert_tokenizer.sep_token = '<|endoftext|>'
            bert_tokenizer.pad_token = '<|pad|>'
            bert_config = AutoConfig.from_pretrained(args.bert_output_dir)
            bert_model = AutoModel.from_pretrained(args.bert_output_dir)
        elif config['emb_class'] in ['t5']:    
            from transformers import T5EncoderModel
            bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_output_dir)
            bert_tokenizer.cls_token = '<s>'
            bert_tokenizer.sep_token = '</s>'
            bert_tokenizer.pad_token = '<pad>'
            bert_config = AutoConfig.from_pretrained(args.bert_output_dir)
            bert_model = T5EncoderModel(bert_config)
        elif config['emb_class'] in ['megatronbert']:    
            from transformers import BertTokenizer, MegatronBertModel
            bert_tokenizer = BertTokenizer.from_pretrained(args.bert_output_dir)
            bert_model = MegatronBertModel.from_pretrained(args.bert_output_dir)
            bert_config = bert_model.config
        else:
            bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_output_dir)
            bert_config = AutoConfig.from_pretrained(args.bert_output_dir)
            bert_model = AutoModel.from_config(bert_config)

        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        if config['enc_class'] == 'densenet-cnn': ModelClass = TextBertDensenetCNN

        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, label_size)

    if args.enable_qat:
        assert args.device == 'cpu'
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        '''
        # fuse if applicable
        # model = torch.quantization.fuse_modules(model, [['']])
        '''
        model = torch.quantization.prepare_qat(model)
        model.eval()
        model.to('cpu')
        logger.info("[Convert to quantized model with device=cpu]")
        model = torch.quantization.convert(model)
    if args.enable_qat_fx:
        import torch.quantization.quantize_fx as quantize_fx
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
        model = quantize_fx.prepare_qat_fx(model, qconfig_dict)
        logger.info("[Convert to quantized model]")
        model = quantize_fx.convert_fx(model)

    if args.enable_diffq:
        quantizer = DiffQuantizer(model)
        config['quantizer'] = quantizer
        quantizer.restore_quantized_state(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(args.device)
    ''' 
    for name, param in model.named_parameters():
        print(name, param.data, param.device, param.requires_grad)
    '''
    logger.info("[model] :\n{}".format(model.__str__()))
    logger.info("[Model loaded]")
    return model

def convert_onnx(config, torch_model, x):
    args = config['args']
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
                          args.onnx_path,                # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=args.onnx_opset, # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          verbose=True,
                          input_names=input_names,      # the model's input names
                          output_names=output_names,    # the model's output names
                          dynamic_axes=dynamic_axes)    # variable length axes

def quantize_onnx(onnx_path, quantized_onnx_path):
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize

    onnx_model = onnx.load(onnx_path)

    quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )

    onnx.save_model(quantized_model, quantized_onnx_path)

def check_onnx(config):
    args = config['args']
    import onnx
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

def build_onnx_input(config, ort_session, x):
    args = config['args']
    x = to_numpy(x)
    if config['emb_class'] == 'glove':
        ort_inputs = {ort_session.get_inputs()[0].name: x}
    else:
        if config['emb_class'] in ['roberta', 'distilbert', 'bart', 'ibert', 't5']:
            ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                          ort_session.get_inputs()[1].name: x[1]}
        else:
            ort_inputs = {ort_session.get_inputs()[0].name: x[0],
                          ort_session.get_inputs()[1].name: x[1],
                          ort_session.get_inputs()[2].name: x[2]}
    return ort_inputs

# ---------------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------------- #

def get_soft_label(args, pred):
    soft_label = None
    if args.entropy_threshold != -1: # given threshold
        from scipy.stats import entropy
        from scipy.special import softmax
        prob = softmax(pred)
        ent = entropy(prob, base=2)
        if ent <= args.entropy_threshold:
            soft_label = ['%.6f' % p for p in pred]
    else:
        soft_label = ['%.6f' % p for p in pred]
    return soft_label

def write_prediction(args, preds, labels):
    # load test data
    tot_num_line = sum(1 for _ in open(args.test_path, 'r')) 
    with open(args.test_path, 'r', encoding='utf-8') as f:
        data = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens) == 2: # single sentence
                sent_a = tokens[0]
                sent_b = None
            if len(tokens) == 3: # sentence pair
                sent_a = tokens[0]
                sent_b = tokens[1]
            label = tokens[-1]
            data.append((sent_a, sent_b, label))
    # write prediction
    try:
        pred_path = args.test_path + '.pred'
        with open(pred_path, 'w', encoding='utf-8') as f:
            for entry, pred in zip(data, preds):
                sent_a, sent_b, label = entry
                text = sent_a
                if sent_b: text = sent_a + '\t' + sent_b
                if args.augmented:
                    soft_label = get_soft_label(args, pred)
                    if soft_label:
                        if args.hard_labeling:
                            pred_id = np.argmax(pred)
                            pred_label = labels[pred_id]
                            f.write(text + '\t' + pred_label + '\n')
                        else:
                            f.write(text + '\t' + ' '.join(logits) + '\n')
                else:
                    pred_id = np.argmax(pred)
                    pred_label = labels[pred_id]
                    f.write(text + '\t' + label + '\t' + pred_label + '\n')
    except Exception as e:
        logger.warn(str(e))

def prepare_datasets(config):
    args = config['args']
    if config['emb_class'] == 'glove':
        DatasetClass = GloveDataset
    else:
        DatasetClass = BertDataset
    test_loader = prepare_dataset(config, args.data_path, DatasetClass, sampling=False, num_workers=1)
    return test_loader

def evaluate(args):
    # set config
    config = load_config(args)
    if args.num_threads > 0: torch.set_num_threads(args.num_threads)
    config['args'] = args
    logger.info("%s", config)

    # set path
    set_path(config)

    # prepare test dataset
    test_loader = prepare_datasets(config)
 
    # load pytorch model checkpoint
    checkpoint = load_checkpoint(args.model_path, device=args.device)

    # prepare model and load parameters
    model = load_model(config, checkpoint)
    model.eval()

    # convert to onnx
    if args.convert_onnx:
        (x, y) = next(iter(test_loader))
        x = to_device(x, args.device)
        y = to_device(y, args.device)
        convert_onnx(config, model, x)
        check_onnx(config)
        logger.info("[ONNX model saved] :{}".format(args.onnx_path))
        # quantize onnx
        if args.quantize_onnx:
            quantize_onnx(args.onnx_path, args.quantized_onnx_path)
            logger.info("[Quantized ONNX model saved] : {}".format(args.quantized_onnx_path))
        return

    # load onnx model for using onnxruntime
    if args.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = args.num_threads
        sess_options.intra_op_num_threads = args.num_threads
        ort_session = ort.InferenceSession(args.onnx_path, sess_options=sess_options)

    # enable to use dynamic quantized model (pytorch>=1.3.0)
    if args.enable_dqm and args.device == 'cpu':
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
            x = to_device(x, args.device)
            y = to_device(y, args.device)

            if args.enable_ort:
                ort_inputs = build_onnx_input(config, ort_session, x)
                logits = ort_session.run(None, ort_inputs)[0]
                logits = to_device(torch.tensor(logits), args.device)
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
            if args.num_examples != 0 and total_examples >= args.num_examples:
                logger.info("[Stop Evaluation] : up to the {} examples".format(total_examples))
                break
            duration_time = float((time.time()-start_time)*1000)
            if i != 0: total_duration_time += duration_time
            '''
            logger.info("[Elapsed Time] : {}ms".format(duration_time))
            '''
    # generate report
    labels = config['labels']
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
    write_prediction(args, preds, labels)
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
    args = config['args']
    if config['emb_class'] == 'glove':
        vocab = load_vocab(args.vocab_path)
        tokenizer = Tokenizer(vocab, config)
    else:
        tokenizer = model.bert_tokenizer
    return tokenizer

def encode_text(config, tokenizer, sent_a, sent_b):
    if config['emb_class'] == 'glove':
        # not yet supporting for sentence pair(TODO)
        text = sent_a
        tokens = text.split()
        # kernel size can't be greater than actual input size,
        # we should pad the sequence up to the maximum kernel size + 1.
        min_seq_size = 10
        ids = tokenizer.convert_tokens_to_ids(tokens, pad_sequence=False, min_seq_size=min_seq_size)
        x = torch.tensor([ids])
        # x : [batch_size, variable size]
        # batch size: 1
    else:
        if sent_b:
            inputs = tokenizer.encode_plus(sent_a, sent_b, add_special_tokens=True, return_tensors='pt')
        else:
            inputs = tokenizer.encode_plus(sent_a, add_special_tokens=True, return_tensors='pt')
        if config['emb_class'] in ['roberta', 'bart', 'distilbert', 'ibert', 't5']:
            x = [inputs['input_ids'], inputs['attention_mask']]
            # x[0], x[1] : [batch_size, variable size]
        else:
            x = [inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']]
            # x[0], x[1], x[2] : [batch_size, variable size]
        # batch size: 1
    return x

def inference(args):
    # set config
    config = load_config(args)
    if args.num_threads > 0: torch.set_num_threads(args.num_threads)
    config['args'] = args

    # set path: args.embedding_path, args.vocab_path, args.label_path
    set_path(config)
 
    # load pytorch model checkpoint
    checkpoint = load_checkpoint(args.model_path, device=args.device)

    # prepare model and load parameters
    model = load_model(config, checkpoint)
    model.eval()

    # load onnx model for using onnxruntime
    if args.enable_ort:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = args.num_threads
        sess_options.intra_op_num_threads = args.num_threads
        ort_session = ort.InferenceSession(args.onnx_path, sess_options=sess_options)

    # enable to use dynamic quantized model (pytorch>=1.3.0)
    if args.enable_dqm and args.device == 'cpu':
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(model)

    # prepare tokenizer
    tokenizer = prepare_tokenizer(config, model)

    # prepare labels
    labels = config['labels']

    # inference
    f_out = open(args.test_path + '.inference', 'w', encoding='utf-8')
    total_examples = 0
    total_duration_time = 0.0
    with torch.no_grad(), open(args.test_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            start_time = time.time()
            tokens = line.strip().split('\t')
            if len(tokens) == 2: # single sentence
                sent_a = tokens[0]
                sent_b = None
                text = sent_a
            if len(tokens) == 3: # sentence pair
                sent_a = tokens[0]
                sent_b = tokens[1]
                text = sent_a + '\t' + sent_b
            label = tokens[-1]
            y_raw = label
            x = encode_text(config, tokenizer, sent_a, sent_b)
            x = to_device(x, args.device)
            if args.enable_ort:
                ort_inputs = build_onnx_input(config, ort_session, x)
                logits = ort_session.run(None, ort_inputs)[0]
                logits = to_device(torch.tensor(logits), args.device)
            else:
                logits = model(x)

            predicted = logits.argmax(1)
            predicted = to_numpy(predicted)[0]
            predicted_raw = labels[predicted]
            f_out.write(text + '\t' + y_raw + '\t' + predicted_raw + '\n')
            total_examples += 1
            if args.num_examples != 0 and total_examples >= args.num_examples:
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
                        help="Set this flag to generate augmented.raw.pred(augmented.txt) for training.")
    parser.add_argument('--entropy_threshold', type=float, default=-1,
                        help="Filtering out soft labeled samples of which entropy is above the threshold."
                             "default value is negative so that filtering will not be applied.")
    parser.add_argument('--hard_labeling', action='store_true',
                        help="Hard labeling instead of soft labeling.")
    # for BERT
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The checkpoint directory of fine-tuned BERT model.")
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
    # for Dynamic Quantization
    parser.add_argument('--enable_dqm', action='store_true',
                        help="Set this flag to use dynamic quantized model.")
    # for Inference
    parser.add_argument('--enable_inference', action='store_true',
                        help="Set this flag to inference for raw input text.")
    # for QAT
    parser.add_argument('--enable_qat', action='store_true',
                        help="Set this flag to use the model by quantization aware training.")
    parser.add_argument('--enable_qat_fx', action='store_true',
                        help="Set this flag for quantization aware training using fx graph mode.")
    # for DiffQ
    parser.add_argument('--enable_diffq', action='store_true',
                        help="Set this flag to use diffq(Differentiable Model Compression).")

    args = parser.parse_args()

    if args.enable_inference:
        inference(args)
    else:
        evaluate(args) 

if __name__ == '__main__':
    main()
