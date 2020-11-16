from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.autograd.profiler as profiler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass
import numpy as np
import random
import json
from tqdm import tqdm

from util    import load_config, to_device, to_numpy
from model   import TextGloveGNB, TextGloveCNN, TextGloveDensenetCNN, TextGloveDensenetDSA, TextBertCNN, TextBertCLS
from dataset import prepare_dataset, GloveDataset, BertDataset
from early_stopping import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from datasets.metric import temp_seed 

import optuna
import nni

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler('./train.log')
logger.addHandler(fileHandler)

def train_epoch(model, config, train_loader, val_loader, epoch_i, best_eval_measure):
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    writer = config['writer']
    scaler = config['scaler']
    opt = config['opt']

    if opt.criterion == 'MSELoss':
        criterion = torch.nn.MSELoss(reduction='sum').to(opt.device)
    elif opt.criterion == 'KLDivLoss':
        criterion = torch.nn.KLDivLoss(reduction='sum').to(opt.device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    # train one epoch
    total_loss = 0
    avg_loss = 0
    best_eval_loss = float('inf')
    best_eval_acc = 0
    total_examples = 0
    st_time = time.time()
    optimizer.zero_grad()
    epoch_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch_i}")
    for local_step, (x,y) in enumerate(epoch_iterator):
        model.train()
        global_step = (len(train_loader) * epoch_i) + local_step
        x = to_device(x, opt.device)
        y = to_device(y, opt.device)
        with autocast(enabled=opt.use_amp):
            if opt.use_profiler:
                with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                    output = model(x)
                print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            else:
                output = model(x)
            if opt.criterion == 'KLDivLoss':
                loss = criterion(F.log_softmax(output, dim=1), y)
            else:
                loss = criterion(output, y)
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
        # back-propagation - begin
        scaler.scale(loss).backward()
        if (local_step + 1) % opt.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if opt.use_transformers_optimizer: scheduler.step()
            curr_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            epoch_iterator.set_description(f"Epoch {epoch_i}, local_step: {local_step}, loss: {loss:.3f}, curr_lr: {curr_lr:.7f}")
            if opt.eval_and_save_steps > 0 and global_step != 0 and global_step % opt.eval_and_save_steps == 0:
                # evaluate
                eval_loss, eval_acc = evaluate(model, config, val_loader)
                if best_eval_loss > eval_loss: best_eval_loss = eval_loss
                if best_eval_acc < eval_acc: best_eval_acc = eval_acc
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
                        logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_acc))
                        save_model(config, model)
                        # save finetuned bert model/config/tokenizer
                        if config['emb_class'] not in ['glove']:
                            if not os.path.exists(opt.bert_output_dir):
                                os.makedirs(opt.bert_output_dir)
                            model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                            model.bert_model.save_pretrained(opt.bert_output_dir)
        # back-propagation - end
        cur_examples = y.size(0)
        total_examples += cur_examples
        total_loss += (loss.item() * cur_examples)
        if writer: writer.add_scalar('Loss/train', loss.item(), global_step)
    avg_loss = total_loss / total_examples

    # evaluate at the end of epoch
    eval_loss, eval_acc = evaluate(model, config, val_loader)
    if best_eval_loss > eval_loss: best_eval_loss = eval_loss
    if best_eval_acc < eval_acc: best_eval_acc = eval_acc
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
            logger.info("[Best model saved] : {}, {}".format(eval_loss, eval_acc))
            save_model(config, model)
            # save finetuned bert model/config/tokenizer
            if config['emb_class'] not in ['glove']:
                if not os.path.exists(opt.bert_output_dir):
                    os.makedirs(opt.bert_output_dir)
                model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                model.bert_model.save_pretrained(opt.bert_output_dir)

    curr_time = time.time()
    elapsed_time = (curr_time - st_time) / 60
    st_time = curr_time
    logger.info('{:3d} epoch | {:5d}/{:5d} | train loss : {:6.3f} | {:5.2f} min elapsed'.\
            format(epoch_i, local_step+1, len(train_loader), avg_loss, elapsed_time)) 

    return best_eval_loss, best_eval_acc, best_eval_measure
 
def evaluate(model, config, val_loader):
    opt = config['opt']
    total_loss = 0.
    total_examples = 0 
    correct = 0
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    preds = None
    ys    = None
    with torch.no_grad():
        for i, (x,y) in tqdm(enumerate(val_loader), total=len(val_loader)):
            model.eval()
            x = to_device(x, opt.device)
            y = to_device(y, opt.device)
            logits = model(x)
            loss = criterion(logits, y)

            if preds is None:
                preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)
            predicted = logits.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = y.size(0)
            total_loss += (loss.item() * cur_examples) 
            total_examples += cur_examples
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
    cur_loss = total_loss / total_examples
    cur_acc  = correct / total_examples
    return cur_loss, cur_acc

def save_model(config, model, save_path=None):
    opt = config['opt']
    checkpoint_path = opt.save_path
    if save_path: checkpoint_path = save_path
    with open(checkpoint_path, 'wb') as f:
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

def prepare_datasets(config, hp_search_bsz=None):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        DatasetClass = GloveDataset
    else:
        DatasetClass = BertDataset
    train_loader = prepare_dataset(config,
        opt.train_path,
        DatasetClass,
        sampling=True,
        num_workers=2,
        hp_search_bsz=hp_search_bsz)
    valid_loader = prepare_dataset(config,
        opt.valid_path,
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
    # prepare model
    if config['emb_class'] == 'glove':
        if config['enc_class'] == 'gnb':
            model = TextGloveGNB(config, opt.embedding_path, opt.label_path)
        if config['enc_class'] == 'cnn':
            model = TextGloveCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=emb_non_trainable)
        if config['enc_class'] == 'densenet-cnn':
            model = TextGloveDensenetCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=emb_non_trainable)
        if config['enc_class'] == 'densenet-dsa':
            model = TextGloveDensenetDSA(config, opt.embedding_path, opt.label_path, emb_non_trainable=emb_non_trainable)
    else:
        model_name_or_path = opt.bert_model_name_or_path
        if bert_model_name_or_path: model_name_or_path = bert_model_name_or_path
        if config['emb_class'] == 'funnel':
            from transformers import FunnelTokenizer, FunnelConfig, FunnelBaseModel
            bert_tokenizer = FunnelTokenizer.from_pretrained(model_name_or_path)
            bert_model = FunnelBaseModel.from_pretrained(model_name_or_path,
                                                         from_tf=bool(".ckpt" in model_name_or_path))
        else:
            from transformers import AutoTokenizer, AutoConfig, AutoModel
            bert_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            bert_model = AutoModel.from_pretrained(model_name_or_path,
                                                   from_tf=bool(".ckpt" in model_name_or_path))
        bert_config = bert_model.config
        # bert model reduction
        reduce_bert_model(config, bert_model, bert_config)
        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path, feature_based=opt.bert_use_feature_based)
    model.to(opt.device)
    logger.info("[model] :\n{}".format(model.__str__()))
    logger.info("[model prepared]")
    return model

def prepare_osws(config, model, train_loader, hp_search_optuna_lr=None):
    opt = config['opt']
    lr = opt.lr
    # for optuna
    if hp_search_optuna_lr: lr = hp_search_optuna_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=opt.adam_epsilon, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.lr_decay_rate)
    if opt.use_transformers_optimizer:
        from transformers import AdamW, get_linear_schedule_with_warmup
        num_training_steps_for_epoch = len(train_loader) // opt.gradient_accumulation_steps
        num_training_steps = num_training_steps_for_epoch * opt.epoch
        num_warmup_steps = num_training_steps_for_epoch * opt.warmup_epoch
        logger.info("(num_training_steps_for_epoch, num_training_steps, num_warmup_steps): ({}, {}, {})".\
            format(num_training_steps_for_epoch, num_training_steps, num_warmup_steps))        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': opt.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=opt.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None
    scaler = GradScaler()
    logger.info("[Creating optimizer, scheduler, summary writer, scaler]")
    return optimizer, scheduler, writer, scaler

def train(opt):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    # set etc
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)
  
    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    with temp_seed(opt.seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer, scaler
        optimizer, scheduler, writer, scaler = prepare_osws(config, model, train_loader)
        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['writer'] = writer
        config['scaler'] = scaler

        # training
        early_stopping = EarlyStopping(logger, patience=opt.patience, measure=opt.measure, verbose=1)
        local_worse_epoch = 0
        prev_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
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
            if opt.measure == 'loss': is_best = eval_measure < best_eval_measure
            else: is_best = eval_measure > best_eval_measure
            if is_best:
                best_eval_measure = eval_measure
                early_stopping.reset(best_eval_measure)
            early_stopping.status()
            # begin: scheduling, apply rate decay at the measure(ex, loss) getting worse for the number of deacy epoch steps.
            if opt.measure == 'loss': getting_worse = prev_eval_measure <= eval_measure
            else: getting_worse = prev_eval_measure >= eval_measure
            if getting_worse:
                local_worse_epoch += 1
            else:
                local_worse_epoch = 0
            logger.info('Scheduler: local_worse_epoch / opt.lr_decay_epoch = %d / %d' % (local_worse_epoch, opt.lr_decay_epoch))
            if not opt.use_transformers_optimizer and \
               epoch_i > opt.warmup_epoch and \
               (local_worse_epoch >= opt.lr_decay_epoch or early_stopping.step() > opt.lr_decay_epoch):
                scheduler.step()
                local_worse_epoch = 0
            prev_eval_measure = eval_measure
            # end: scheduling
        # for nni
        if opt.hp_search_nni:
            nni.report_final_result(eval_acc)
            logger.info('[Final result] : %g', eval_acc)
            logger.info('[Send final result done]')

# for optuna, global for passing opt 
gopt = None

def hp_search_optuna(trial: optuna.Trial):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    global gopt
    opt = gopt
    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

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
        # create optimizer, scheduler, summary writer, scaler
        optimizer, scheduler, writer, scaler = prepare_osws(config, model, train_loader, hp_search_optuna_lr=lr)
        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['writer'] = writer
        config['scaler'] = scaler

        early_stopping = EarlyStopping(logger, patience=opt.patience, measure=opt.measure, verbose=1)
        best_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
        for epoch in range(epochs):
            eval_loss, eval_acc, best_eval_measure = train_epoch(model, config, train_loader, valid_loader, epoch, best_eval_measure)

            if opt.measure == 'loss': eval_measure = eval_loss 
            else: eval_measure = eval_acc
            # early stopping
            if early_stopping.validate(eval_measure, measure=opt.measure): break
            if opt.measure == 'loss': is_best = eval_measure < best_eval_measure
            else: is_best = eval_measure > best_eval_measure
            if is_best:
                best_eval_measure = eval_measure
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
    parser.add_argument('--epoch', type=int, default=64)
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=1.0, help="Disjoint with --use_transformers_optimizer")
    parser.add_argument('--lr_decay_epoch', type=float, default=2, help="Number of decay epoch to be paitent. disjoint with --use_transformers_optimizer")
    parser.add_argument('--warmup_epoch', type=int, default=4,  help="Number of warmup epoch steps")
    parser.add_argument('--patience', default=7, type=int, help="Max number of epoch to be patient for early stopping.")
    parser.add_argument('--save_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--embedding_trainable', action='store_true', help="Set word embedding(Glove) trainable")
    parser.add_argument('--use_transformers_optimizer', action='store_true', help="Use transformers AdamW, get_linear_schedule_with_warmup.")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision.")
    parser.add_argument('--use_profiler', action='store_true', help="Use profiler.")
    parser.add_argument('--measure', type=str, default='loss', help="Evaluation measure, 'loss' | 'accuracy', default 'loss'.")
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help="training objective, 'CrossEntropyLoss' | 'MSELoss' | 'KLDivLoss', default 'CrossEntropyLoss'")
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
