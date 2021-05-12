import sys
import os
import argparse
import time
import pdb
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from diffq import DiffQuantizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass
import numpy as np
import random
import json
from tqdm import tqdm

from util    import load_checkpoint, load_config, load_label, to_device
from model   import TextGloveGNB, TextGloveCNN, TextGloveDensenetCNN, TextGloveDensenetDSA, TextBertCNN, TextBertCLS
from dataset import prepare_dataset, GloveDataset, BertDataset
from early_stopping import EarlyStopping
from label_smoothing import LabelSmoothingCrossEntropy
from sklearn.metrics import classification_report, confusion_matrix
from datasets.metric import temp_seed 

import optuna
import nni

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler('./train.log')
logger.addHandler(fileHandler)

def train_epoch(model, config, train_loader, valid_loader, epoch_i, best_eval_measure):
    opt = config['opt']
    accelerator = config['accelerator'] 

    optimizer = config['optimizer']
    scheduler = config['scheduler']
    writer = config['writer']

    if opt.criterion == 'MSELoss':
        criterion = torch.nn.MSELoss(reduction='sum')
    elif opt.criterion == 'KLDivLoss':
        criterion = torch.nn.KLDivLoss(reduction='sum')
    elif opt.criterion == 'LabelSmoothingCrossEntropy':
        criterion = LabelSmoothingCrossEntropy(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # train one epoch
    total_loss = 0
    avg_loss = 0
    local_best_eval_loss = float('inf')
    local_best_eval_acc = 0
    total_examples = 0
    st_time = time.time()
    optimizer.zero_grad()
    epoch_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch_i}")
    for local_step, (x,y) in enumerate(epoch_iterator):
        model.train()
        global_step = (len(train_loader) * epoch_i) + local_step
        output = model(x)
        if opt.criterion == 'KLDivLoss':
            loss = criterion(F.log_softmax(output, dim=1), y)
        else:
            loss = criterion(output, y)
        if opt.enable_diffq:
            quantizer = config['quantizer']
            loss = loss  + opt.diffq_penalty * quantizer.model_size()
        if opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps

        # back-propagation - begin
        accelerator.backward(loss)
        if (local_step + 1) % opt.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            epoch_iterator.set_description(f"Epoch {epoch_i}, local_step: {local_step}, loss: {loss:.3f}, curr_lr: {curr_lr:.7f}")
            if accelerator.is_main_process and opt.eval_and_save_steps > 0 and global_step != 0 and global_step % opt.eval_and_save_steps == 0:
                # evaluate
                eval_loss, eval_acc = evaluate(model, config, valid_loader)
                if local_best_eval_loss > eval_loss: local_best_eval_loss = eval_loss
                if local_best_eval_acc < eval_acc: local_best_eval_acc = eval_acc
                if writer:
                    writer.add_scalar('Loss/valid', eval_loss, global_step)
                    writer.add_scalar('Acc/valid', eval_acc, global_step)
                    writer.add_scalar('LearningRate/train', curr_lr, global_step)
                if opt.measure == 'loss': eval_measure = eval_loss 
                else: eval_measure = eval_acc
                if opt.measure == 'loss': is_best = eval_measure < best_eval_measure
                else: is_best = eval_measure > best_eval_measure
                if is_best:
                    best_eval_measure = eval_measure
                    if opt.save_path and not opt.hp_search_optuna and not opt.hp_search_nni:
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(config, unwrapped_model, valid_loader=valid_loader)
                        logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_acc))
                        # save finetuned bert model/config/tokenizer
                        if config['emb_class'] not in ['glove'] and not (config['emb_class'] == 'bart' and config['use_kobart']):
                            if not os.path.exists(opt.bert_output_dir):
                                os.makedirs(opt.bert_output_dir)
                            unwrapped_model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                            unwrapped_model.bert_model.save_pretrained(opt.bert_output_dir)
        # back-propagation - end
        cur_examples = y.size(0)
        total_examples += cur_examples
        total_loss += (loss.item() * cur_examples)
        if writer: writer.add_scalar('Loss/train', loss.item(), global_step)
    avg_loss = total_loss / total_examples

    # evaluate at the end of epoch
    if accelerator.is_main_process:
        eval_loss, eval_acc = evaluate(model, config, valid_loader)
        if local_best_eval_loss > eval_loss: local_best_eval_loss = eval_loss
        if local_best_eval_acc < eval_acc: local_best_eval_acc = eval_acc
        if writer:
            writer.add_scalar('Loss/valid', eval_loss, global_step)
            writer.add_scalar('Acc/valid', eval_acc, global_step)
            writer.add_scalar('LearningRate/train', curr_lr, global_step)
        if opt.measure == 'loss': eval_measure = eval_loss 
        else: eval_measure = eval_acc
        if opt.measure == 'loss': is_best = eval_measure < best_eval_measure
        else: is_best = eval_measure > best_eval_measure
        if is_best:
            best_eval_measure = eval_measure
            if opt.save_path and not opt.hp_search_optuna and not opt.hp_search_nni:
                unwrapped_model = accelerator.unwrap_model(model)
                save_model(config, unwrapped_model, valid_loader=valid_loader)
                logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_acc))
                # save finetuned bert model/config/tokenizer
                if config['emb_class'] not in ['glove'] and not (config['emb_class'] == 'bart' and config['use_kobart']):
                    if not os.path.exists(opt.bert_output_dir):
                        os.makedirs(opt.bert_output_dir)
                    unwrapped_model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                    unwrapped_model.bert_model.save_pretrained(opt.bert_output_dir)

    curr_time = time.time()
    elapsed_time = (curr_time - st_time) / 60
    st_time = curr_time
    logs = {'epoch': epoch_i,
           'local_step': local_step+1,
           'epoch_step': len(train_loader),
           'avg_loss': avg_loss,
           'local_best_eval_loss': local_best_eval_loss,
           'local_best_eval_acc': local_best_eval_acc,
           'best_eval_measure': best_eval_measure,
           'elapsed_time': elapsed_time
    }
    logger.info(json.dumps(logs, indent=4, ensure_ascii=False, sort_keys=True))

    return local_best_eval_loss, local_best_eval_acc, best_eval_measure
 
def evaluate(model, config, valid_loader, eval_device=None):
    opt = config['opt']

    total_loss = 0.
    total_examples = 0 
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    preds = None
    ys    = None
    with torch.no_grad():
        iterator = tqdm(valid_loader, total=len(valid_loader), desc=f"Evaluate")
        for i, (x,y) in enumerate(iterator):
            if eval_device:
                x = to_device(x, opt.device)
                y = to_device(y, opt.device)
            model.eval()
            logits = model(x)
            loss = criterion(logits, y)
            # softmax after computing cross entropy loss
            logits = torch.softmax(logits, dim=-1)
            logits = logits.cpu().numpy()
            y = y.cpu().numpy()
            if preds is None:
                preds = logits
                ys = y
            else:
                preds = np.append(preds, logits, axis=0)
                ys = np.append(ys, y, axis=0)
            predicted = np.argmax(logits, axis=1)
            correct += np.sum(np.equal(predicted, y).astype(int))
            cur_examples = y.size
            total_loss += (loss.item() * cur_examples) 
            total_examples += cur_examples
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
    cur_loss = total_loss / total_examples
    cur_acc  = correct / total_examples
    return cur_loss, cur_acc

def save_model(config, model, valid_loader=None, save_path=None):
    opt = config['opt']
    checkpoint_path = opt.save_path
    if save_path: checkpoint_path = save_path
    with open(checkpoint_path, 'wb') as f:
        if opt.enable_qat:
            '''
            for name, param in model.named_parameters():
                print(name, param.data, param.device, param.requires_grad)
            '''
            import copy
            model_to_quantize = copy.deepcopy(model)
            model_to_quantize.eval()
            model_to_quantize.to('cpu')
            logger.info("[Convert to quantized model with device=cpu]")
            # torch.quantization.convert() only supports CPU.
            quantized_model = torch.quantization.convert(model_to_quantize)
            logger.info("[Evaluate quantized model with device=cpu]")
            evaluate(quantized_model, config, valid_loader, eval_device='cpu')
            checkpoint = quantized_model.state_dict()
        elif opt.enable_qat_fx:
            import torch.quantization.quantize_fx as quantize_fx
            import copy
            model_to_quantize = copy.deepcopy(model)
            model_to_quantize.eval()
            model_to_quantize.to('cpu')
            logger.info("[Convert to quantized model with device=cpu]")
            quantized_model = quantize_fx.convert_fx(model_to_quantize)
            logger.info("[Evaluate quantized model with device=cpu]")
            evaluate(quantized_model, config, valid_loader, eval_device='cpu')
            checkpoint = quantized_model.state_dict()
        elif opt.enable_diffq:
            quantizer = config['quantizer']
            logger.info("true naive model size: {}".format(quantizer.true_model_size()))
            logger.info("compressed model size: {}".format(quantizer.compressed_model_size()))
            checkpoint = quantizer.get_quantized_state()
        else:
            checkpoint = model.state_dict()
        torch.save(checkpoint,f)

def set_path(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        if opt.augmented:
            opt.train_path = os.path.join(opt.data_dir, 'augmented.txt.ids')
        else:
            opt.train_path = os.path.join(opt.data_dir, 'train.txt.ids')
        opt.valid_path = os.path.join(opt.data_dir, 'valid.txt.ids')
    else:
        if opt.augmented:
            opt.train_path = os.path.join(opt.data_dir, 'augmented.txt.fs')
        else:
            opt.train_path = os.path.join(opt.data_dir, 'train.txt.fs')
        opt.valid_path = os.path.join(opt.data_dir, 'valid.txt.fs')
    opt.label_path     = os.path.join(opt.data_dir, opt.label_filename)
    opt.embedding_path = os.path.join(opt.data_dir, opt.embedding_filename)

def prepare_datasets(config, hp_search_bsz=None, train_path=None, valid_path=None):
    opt = config['opt']
    default_train_path = opt.train_path
    default_valid_path = opt.valid_path
    if train_path: default_train_path = train_path
    if valid_path: default_valid_path = valid_path
    if config['emb_class'] == 'glove':
        DatasetClass = GloveDataset
    else:
        DatasetClass = BertDataset
    train_loader = prepare_dataset(config,
        default_train_path,
        DatasetClass,
        sampling=True,
        num_workers=2,
        hp_search_bsz=hp_search_bsz)
    valid_loader = prepare_dataset(config,
        default_valid_path,
        DatasetClass,
        sampling=False,
        num_workers=2,
        batch_size=opt.eval_batch_size)
    return train_loader, valid_loader

def get_bert_embed_layer_list(config, bert_model):
    opt = config['opt']
    embed_list = list(bert_model.embeddings.parameters())
    # note that 'distilbert' has no encoder.layer, so don't use bert_remove_layers for distilbert.
    layer_list = bert_model.encoder.layer
    return embed_list, layer_list

def reduce_bert_model(config, bert_model, bert_config):
    opt = config['opt']
    remove_layers = opt.bert_remove_layers
    # drop layers
    if remove_layers is not "":
        embed_list, layer_list = get_bert_embed_layer_list(config, bert_model)
        layer_indexes = [int(x) for x in remove_layers.split(",")]
        layer_indexes.sort(reverse=True)
        for layer_idx in layer_indexes:
            if layer_idx < 0 or layer_idx >= bert_config.num_hidden_layers: continue
            del(layer_list[layer_idx])
            logger.info("[layer removed] : %s" % (layer_idx))
        if len(layer_indexes) > 0:
            bert_config.num_hidden_layers = len(layer_list)

def prepare_model(config, bert_model_name_or_path=None):
    opt = config['opt']
    emb_non_trainable = not opt.embedding_trainable
    labels = load_label(opt.label_path)
    label_size = len(labels)
    config['labels'] = labels
    # prepare model
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'gnb':
            model = TextGloveGNB(config, opt.embedding_path, label_size)
        if config['enc_class'] == 'cnn':
            model = TextGloveCNN(config, opt.embedding_path, label_size, emb_non_trainable=emb_non_trainable)
        if config['enc_class'] == 'densenet-cnn':
            model = TextGloveDensenetCNN(config, opt.embedding_path, label_size, emb_non_trainable=emb_non_trainable)
        if config['enc_class'] == 'densenet-dsa':
            model = TextGloveDensenetDSA(config, opt.embedding_path, label_size, emb_non_trainable=emb_non_trainable)
    else:
        model_name_or_path = opt.bert_model_name_or_path
        if bert_model_name_or_path: model_name_or_path = bert_model_name_or_path
        
        if config['emb_class'] == 'bart' and config['use_kobart']:
            from transformers import BartModel
            from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
            bert_tokenizer = get_kobart_tokenizer()
            bert_tokenizer.cls_token = '<s>'
            bert_tokenizer.sep_token = '</s>'
            bert_tokenizer.pad_token = '<pad>'
            bert_model = BartModel.from_pretrained(get_pytorch_kobart_model())
        else:
            from transformers import AutoTokenizer, AutoConfig, AutoModel
            bert_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if config['emb_class'] == 'gpt': 
                bert_tokenizer.cls_token = '<s>'
                bert_tokenizer.sep_token = '</s>'
                bert_tokenizer.pad_token = '<pad>'
                bert_tokenizer.unk_token = '<unk>'
            bert_model = AutoModel.from_pretrained(model_name_or_path,
                                                   from_tf=bool(".ckpt" in model_name_or_path))

        bert_config = bert_model.config
        # bert model reduction
        reduce_bert_model(config, bert_model, bert_config)
        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, label_size, feature_based=opt.bert_use_feature_based)
    if opt.restore_path:
        checkpoint = load_checkpoint(opt.restore_path)
        model.load_state_dict(checkpoint)
    if opt.enable_qat:
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        '''
        # fuse if applicable
        # model = torch.quantization.fuse_modules(model, [['']])
        '''
        model = torch.quantization.prepare_qat(model)
    if opt.enable_qat_fx:
        import torch.quantization.quantize_fx as quantize_fx
        model.train()
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
        model = quantize_fx.prepare_qat_fx(model, qconfig_dict)

    logger.info("[model] :\n{}".format(model.__str__()))
    logger.info("[model prepared]")
    return model

def prepare_others(config, model, data_loader, lr=None, weight_decay=None):
    opt = config['opt']
    accelerator = None
    if 'accelerator' in config: accelerator = config['accelerator']

    default_lr = opt.lr
    if lr: default_lr = lr
    default_weight_decay = opt.weight_decay
    if weight_decay: default_weight_decay = weight_decay

    num_update_steps_per_epoch = math.ceil(len(data_loader) / opt.gradient_accumulation_steps)
    if opt.max_train_steps is None:
        opt.max_train_steps = opt.epoch * num_update_steps_per_epoch
    else:
        opt.epoch = math.ceil(opt.max_train_steps / num_update_steps_per_epoch)
    if opt.num_warmup_steps is None: 
        if opt.warmup_ratio:
            opt.num_warmup_steps = opt.max_train_steps * opt.warmup_ratio
        if opt.warmup_epoch:
            opt.num_warmup_steps = num_update_steps_per_epoch * opt.warmup_epoch

    logger.info(f"(num_update_steps_per_epoch, max_train_steps, num_warmup_steps): ({num_update_steps_per_epoch}, {opt.max_train_steps}, {opt.num_warmup_steps})")

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': default_weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=default_lr,
                      eps=opt.adam_epsilon,
                      correct_bias=False)

    if opt.enable_diffq:
        quantizer = DiffQuantizer(model)
        quantizer.setup_optimizer(optimizer)
        config['quantizer'] = quantizer

    if accelerator:
        model, optimizer = accelerator.prepare(model, optimizer)
        
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=opt.num_warmup_steps,
        num_training_steps=opt.max_train_steps)

    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None
    return model, optimizer, scheduler, writer

def train(opt):

    # set etc
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # create accelerator
    accelerator = Accelerator()
    config['accelerator'] = accelerator
    opt.device = accelerator.device

    # set path
    set_path(config)
  
    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    with temp_seed(opt.seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer
        model, optimizer, scheduler, writer = prepare_others(config, model, train_loader)
        train_loader = accelerator.prepare(train_loader)
        valid_loader = accelerator.prepare(valid_loader)
        
        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['writer'] = writer

        total_batch_size = opt.batch_size * accelerator.num_processes * opt.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_loader)}")
        logger.info(f"  Num Epochs = {opt.epoch}")
        logger.info(f"  Instantaneous batch size per device = {opt.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {opt.max_train_steps}")

        # training
        early_stopping = EarlyStopping(logger, patience=opt.patience, measure=opt.measure, verbose=1)
        local_worse_epoch = 0
        best_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
        for epoch_i in range(opt.epoch):
            epoch_st_time = time.time()
            eval_loss, eval_acc, best_eval_measure = train_epoch(model, config, train_loader, valid_loader, epoch_i, best_eval_measure)
            # for nni
            if opt.hp_search_nni:
                nni.report_intermediate_result(eval_acc)
                logger.info('[eval_acc] : %g', eval_acc)
                logger.info('[Pipe send intermediate result done]')
            if opt.measure == 'loss': eval_measure = eval_loss 
            else: eval_measure = eval_acc
            # early stopping
            if early_stopping.validate(eval_measure, measure=opt.measure): break
            if eval_measure == best_eval_measure:
                early_stopping.reset(best_eval_measure)
            early_stopping.status()
        # for nni
        if opt.hp_search_nni:
            nni.report_final_result(eval_acc)
            logger.info('[Final result] : %g', eval_acc)
            logger.info('[Send final result done]')

# for optuna, global for passing opt 
gopt = None

def hp_search_optuna(trial: optuna.Trial):

    global gopt
    opt = gopt
    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

    # create accelerator
    accelerator = Accelerator()
    config['accelerator'] = accelerator
    opt.device = accelerator.device

    # set search spaces
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3) # .suggest_float('lr', 1e-6, 1e-3, log=True)
    bsz = trial.suggest_categorical('batch_size', [32, 64, 128])
    seed = trial.suggest_int('seed', 17, 42)
    epochs = trial.suggest_int('epochs', 1, opt.epoch)

    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config, hp_search_bsz=bsz)

    with temp_seed(seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer
        model, optimizer, scheduler, writer = prepare_others(config, model, train_loader, lr=lr)
        train_loader = accelerator.prepare(train_loader)
        valid_loader = accelerator.prepare(valid_loader)

        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['writer'] = writer

        total_batch_size = opt.batch_size * accelerator.num_processes * opt.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_loader)}")
        logger.info(f"  Num Epochs = {opt.epoch}")
        logger.info(f"  Instantaneous batch size per device = {opt.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {opt.max_train_steps}")

        early_stopping = EarlyStopping(logger, patience=opt.patience, measure=opt.measure, verbose=1)
        best_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
        for epoch in range(epochs):
            eval_loss, eval_acc, best_eval_measure = train_epoch(model, config, train_loader, valid_loader, epoch, best_eval_measure)

            if opt.measure == 'loss': eval_measure = eval_loss 
            else: eval_measure = eval_acc
            # early stopping
            if early_stopping.validate(eval_measure, measure=opt.measure): break
            if eval_measure == best_eval_measure:
                early_stopping.reset(best_eval_measure)
            early_stopping.status()

            trial.report(eval_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return eval_acc

def get_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove-cnn.json')
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_filename', type=str, default='embedding.npy')
    parser.add_argument('--label_filename', type=str, default='label.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=64)
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_warmup_steps', type=int, default=None)
    parser.add_argument('--warmup_epoch', type=int, default=0, help="Number of warmup epoch")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="Ratio for warmup over total number of training steps.")
    parser.add_argument('--patience', default=7, type=int, help="Max number of epoch to be patient for early stopping.")
    parser.add_argument('--save_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--restore_path', type=str, default='')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--embedding_trainable', action='store_true', help="Set word embedding(Glove) trainable")
    parser.add_argument('--measure', type=str, default='loss', help="Evaluation measure, 'loss' | 'accuracy', default 'loss'.")
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help="training objective, 'CrossEntropyLoss' | 'LabelSmoothingCrossEntropy' | 'MSELoss' | 'KLDivLoss', default 'CrossEntropyLoss'")
    # for Augmentation
    parser.add_argument('--augmented', action='store_true',
                        help="Set this flag to use augmented.txt.ids or augmented.txt.fs for training.")
    # for BERT
    parser.add_argument('--bert_model_name_or_path', type=str, default='embeddings/bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the BERT model checkpoints will be written.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_remove_layers', type=str, default='',
                        help="Specify layer numbers to remove during finetuning e.g. 8,9,10,11 to remove last 4 layers from BERT base(12 layers)")
    # for Optuna
    parser.add_argument('--hp_search_optuna', action='store_true',
                        help="Set this flag to use hyper-parameter search by Optuna.")
    parser.add_argument('--hp_trials', default=24, type=int,
                        help="Number of trials for hyper-parameter search.")
    # for NNI
    parser.add_argument('--hp_search_nni', action='store_true',
                        help="Set this flag to use hyper-parameter search by NNI.")
    # for QAT
    parser.add_argument('--enable_qat', action='store_true',
                        help="Set this flag for quantization aware training.")
    parser.add_argument('--enable_qat_fx', action='store_true',
                        help="Set this flag for quantization aware training using fx graph mode.")
    # for DiffQ
    parser.add_argument('--enable_diffq', action='store_true',
                        help="Set this flag to use diffq(Differentiable Model Compression).")
    parser.add_argument('--diffq_penalty', default=1e-3, type=float)

    opt = parser.parse_args()
    return opt

def main():
    opt = get_params()
    if opt.hp_search_optuna:
        global gopt
        gopt = opt
        study = optuna.create_study(direction='maximize')
        study.optimize(hp_search_optuna, n_trials=opt.hp_trials)
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        logger.info("%s", str(df))
        logger.info("[study.best_params] : %s", study.best_params)
        logger.info("[study.best_value] : %s", study.best_value)
        logger.info("[study.best_trial] : %s", study.best_trial) # for all, study.trials
    elif opt.hp_search_nni:
        try:
            # get parameters from tuner
            tuner_params = nni.get_next_parameter()
            logger.info('[tuner_params] :')
            logger.info(tuner_params)
            logger.info('[opt] :')
            logger.info(opt)
            # merge to opt
            if tuner_params:
                for k, v in tuner_params.items():
                    assert hasattr(namespace, k), "Args doesn't have received key: %s" % k
                    assert type(getattr(namespace, k)) == type(v), "Received key has different type"
                    setattr(namespace, k, v) 
            logger.info('opt:')
            logger.info(opt)
            train(opt)
        except Exception as exception:
            logger.exception(exception)
            raise
    else:
        train(opt)
   
if __name__ == '__main__':
    main()
