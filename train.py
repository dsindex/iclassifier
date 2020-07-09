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
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

def set_apex_and_distributed(opt):
    if not APEX_AVAILABLE: opt.use_amp = False
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ and opt.use_amp:
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.word_size = torch.distributed.get_world_size()

def train_epoch(model, config, train_loader, val_loader, epoch_i):
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    writer = config['writer']
    opt = config['opt']

    local_rank = opt.local_rank
    use_amp = opt.use_amp
    if opt.augmented:
        criterion = torch.nn.MSELoss(reduction='sum').to(opt.device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    # train one epoch
    model.train()
    optimizer.zero_grad()
    total_loss = 0.
    final_val_loss = 0.
    total_examples = 0
    st_time = time.time()
    for local_step, (x,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        global_step = (len(train_loader) * epoch_i) + local_step
        x = to_device(x, opt.device)
        y = to_device(y, opt.device)
        output = model(x)
        loss = criterion(output, y)
        # back-propagation - begin
        if opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                try:
                    scaled_loss.backward()
                except Exception as e:
                    print(e)
        else:
            loss.backward()
        if (local_step + 1) % opt.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()
            if opt.use_transformers_optimizer: scheduler.step()
            optimizer.zero_grad()
        # back-propagation - end
        cur_examples = y.size(0)
        total_examples += cur_examples
        total_loss += (loss.item() * cur_examples)
        if local_rank == 0 and writer:
            writer.add_scalar('Loss/train', loss.item(), global_step)
    cur_loss = total_loss / total_examples

    # evaluate
    eval_loss, eval_acc = evaluate(model, config, val_loader)
    curr_time = time.time()
    elapsed_time = (curr_time - st_time) / 60
    st_time = curr_time
    curr_lr = scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
    if local_rank == 0:
        logger.info('{:3d} epoch | {:5d}/{:5d} | train loss : {:6.3f}, valid loss {:6.3f}, valid acc {:.4f}| lr :{:7.6f} | {:5.2f} min elapsed'.\
                format(epoch_i, local_step+1, len(train_loader), cur_loss, eval_loss, eval_acc, curr_lr, elapsed_time)) 
        if writer:
            writer.add_scalar('Loss/valid', eval_loss, global_step)
            writer.add_scalar('Acc/valid', eval_acc, global_step)
            writer.add_scalar('LearningRate/train', curr_lr, global_step)
    return eval_loss, eval_acc
 
def evaluate(model, config, val_loader):
    opt = config['opt']
    model.eval()
    total_loss = 0.
    total_examples = 0 
    correct = 0
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    preds = None
    ys    = None
    with torch.no_grad():
        for i, (x,y) in tqdm(enumerate(val_loader), total=len(val_loader)):
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

def save_model(config, model):
    opt = config['opt']
    optimizer = config['optimizer']
    checkpoint_path = opt.save_path
    with open(checkpoint_path, 'wb') as f:
        if opt.use_amp:
            checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }
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
    if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
        if opt.augmented:
            opt.train_path = os.path.join(opt.data_dir, 'augmented.txt.fs')
        else:
            opt.train_path = os.path.join(opt.data_dir, 'train.txt.fs')
        opt.valid_path = os.path.join(opt.data_dir, 'valid.txt.fs')
    opt.label_path     = os.path.join(opt.data_dir, opt.label_filename)
    opt.embedding_path = os.path.join(opt.data_dir, opt.embedding_filename)

def prepare_datasets(config):
    opt = config['opt']
    if config['emb_class'] == 'glove':
        DatasetClass = GloveDataset
    if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
        DatasetClass = BertDataset
    train_loader = prepare_dataset(config,
        opt.train_path,
        DatasetClass,
        sampling=True,
        num_workers=2)
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
            logger.info("%s layer removed" % (layer_idx))
        if len(layer_indexes) > 0:
            bert_config.num_hidden_layers = len(layer_list)

def prepare_model(config):
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
    if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
        from transformers import BertTokenizer, BertConfig, BertModel
        from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel
        from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
        from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
        from transformers import BartConfig, BartTokenizer, BartModel
        from transformers import ElectraConfig, ElectraTokenizer, ElectraModel
        MODEL_CLASSES = {
            "bert": (BertConfig, BertTokenizer, BertModel),
            "distilbert": (DistilBertConfig, DistilBertTokenizer, DistilBertModel),
            "albert": (AlbertConfig, AlbertTokenizer, AlbertModel),
            "roberta": (RobertaConfig, RobertaTokenizer, RobertaModel),
            "bart": (BartConfig, BartTokenizer, BartModel),
            "electra": (ElectraConfig, ElectraTokenizer, ElectraModel),
        }
        Config    = MODEL_CLASSES[config['emb_class']][0]
        Tokenizer = MODEL_CLASSES[config['emb_class']][1]
        Model     = MODEL_CLASSES[config['emb_class']][2]
        bert_tokenizer = Tokenizer.from_pretrained(opt.bert_model_name_or_path,
                                                   do_lower_case=opt.bert_do_lower_case)
        bert_model = Model.from_pretrained(opt.bert_model_name_or_path,
                                           from_tf=bool(".ckpt" in opt.bert_model_name_or_path))
        bert_config = bert_model.config
        # bert model reduction
        reduce_bert_model(config, bert_model, bert_config)
        ModelClass = TextBertCNN
        if config['enc_class'] == 'cls': ModelClass = TextBertCLS
        model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path, feature_based=opt.bert_use_feature_based)
    model.to(opt.device)
    print(model)
    logger.info("[model prepared]")
    return model

def prepare_osw(config, model, train_loader):
    opt = config['opt']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, eps=opt.adam_epsilon, weight_decay=opt.weight_decay)
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    if opt.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if opt.distributed:
        model = DDP(model, delay_allreduce=True)
    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None
    logger.info("[Creating optimizer, scheduler, summary writer...]")
    return optimizer, scheduler, writer

def train(opt):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    # set seed, distributed setting, etc
    set_seed(opt)
    set_apex_and_distributed(opt)
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)
  
    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    # prepare model
    model = prepare_model(config)

    # create optimizer, scheduler, summary writer
    optimizer, scheduler, writer = prepare_osw(config, model, train_loader)
    config['optimizer'] = optimizer
    config['scheduler'] = scheduler
    config['writer'] = writer

    # training
    early_stopping = EarlyStopping(logger, patience=opt.patience, measure=opt.measure, verbose=1)
    local_worse_steps = 0
    prev_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
    best_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
    for epoch_i in range(opt.epoch):
        epoch_st_time = time.time()
        eval_loss, eval_acc = train_epoch(model, config, train_loader, valid_loader, epoch_i)
        if opt.measure == 'loss': eval_measure = eval_loss 
        else: eval_measure = eval_acc
        # early stopping
        if early_stopping.validate(eval_measure, measure=opt.measure): break
        if opt.measure == 'loss': is_best = eval_measure < best_eval_measure
        else: is_best = eval_measure > best_eval_measure
        if opt.local_rank == 0 and is_best:
            best_eval_measure = eval_measure
            if opt.save_path:
                logger.info("[Best model saved] : {:10.6f}".format(best_eval_measure))
                save_model(config, model)
                # save finetuned bert model/config/tokenizer
                if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
                    if not os.path.exists(opt.bert_output_dir):
                        os.makedirs(opt.bert_output_dir)
                    model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                    model.bert_model.save_pretrained(opt.bert_output_dir)
            early_stopping.reset(best_eval_measure)
        early_stopping.status()
        # begin: scheduling, apply rate decay at the measure(ex, loss) getting worse for the number of deacy epoch steps.
        if opt.measure == 'loss': getting_worse = prev_eval_measure <= eval_measure
        else: getting_worse = prev_eval_measure >= eval_measure
        if getting_worse:
            local_worse_steps += 1
        else:
            local_worse_steps = 0
        logger.info('Scheduler: local_worse_steps / opt.lr_decay_steps = %d / %d' % (local_worse_steps, opt.lr_decay_steps))
        if not opt.use_transformers_optimizer and \
           epoch_i > opt.warmup_epoch and \
           (local_worse_steps >= opt.lr_decay_steps or early_stopping.step() > opt.lr_decay_steps):
            scheduler.step()
        prev_eval_measure = eval_measure
        # end: scheduling

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-glove-cnn.json')
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_filename', type=str, default='embedding.npy')
    parser.add_argument('--label_filename', type=str, default='label.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=1.0, help="Disjoint with --use_transformers_optimizer")
    parser.add_argument('--lr_decay_steps', type=float, default=2, help="Number of decay epoch steps to be paitent. disjoint with --use_transformers_optimizer")
    parser.add_argument('--warmup_epoch', type=int, default=4,  help="Number of warmup epoch steps")
    parser.add_argument('--patience', default=7, type=int, help="Max number of epoch to be patient for early stopping.")
    parser.add_argument('--save_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--embedding_trainable', action='store_true', help="Set word embedding(Glove) trainable")
    parser.add_argument('--use_transformers_optimizer', action='store_true', help="Use transformers AdamW, get_linear_schedule_with_warmup.")
    # for Augmentation
    parser.add_argument('--measure', type=str, default='loss', help="Evaluation measure, 'loss' | 'accuracy', default 'loss'.")
    parser.add_argument('--augmented', action='store_true',
                        help="Set this flag to use augmented.txt for training.")
    # for BERT
    parser.add_argument('--bert_model_name_or_path', type=str, default='embeddings/bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_do_lower_case', action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_remove_layers', type=str, default='',
                        help="Specify layer numbers to remove during finetuning e.g. 8,9,10,11 to remove last 4 layers from BERT base(12 layers)")

    opt = parser.parse_args()

    train(opt)
   
if __name__ == '__main__':
    main()
