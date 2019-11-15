import sys
import os
import argparse
import time
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import json
from tqdm import tqdm
from model import TextCNN
from dataset import SnipsDataset

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

def train_epoch(model, config, train_loader, val_loader, epoch_i):
    device = config['device']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    writer = config['writer']
    opt = config['opt']

    local_rank = opt.local_rank
    use_amp = opt.use_amp
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train one epoch
    total_loss = 0.
    final_val_loss = 0.
    total_examples = 0
    st_time = time.time()
    for local_step, (x,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        global_step = (len(train_loader) * (epoch_i-1)) + local_step
        model.train()
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        cur_examples = x.size(0)
        total_examples += cur_examples
        total_loss += (loss.item() * cur_examples)
        if local_rank == 0 and writer:
            writer.add_scalar('Loss/train', loss.item(), global_step)
    cur_loss = total_loss / total_examples

    # evaluate
    eval_loss, eval_acc = evaluate(model, config, val_loader, device)
    curr_time = time.time()
    elapsed_time = (curr_time - st_time) / 60
    st_time = curr_time
    curr_lr = scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
    if local_rank == 0:
        print('{:3d} epoch | {:5d}/{:5d} | train loss : {:6.3f}, valid loss {:6.3f}, valid acc {:.3f}| lr :{:7.6f} | {:5.2f} min elapsed'.\
                format(epoch_i, local_step+1, len(train_loader), cur_loss, eval_loss, eval_acc, curr_lr, elapsed_time)) 
        if writer:
            writer.add_scalar('Loss/valid', eval_loss, global_step)
            writer.add_scalar('Acc/valid', eval_acc, global_step)
            writer.add_scalar('LearningRate/train', curr_lr, global_step)
    return eval_loss
 
def evaluate(model, config, val_loader, device):
    model.eval()
    total_loss = 0.
    total_examples = 0 
    correct = 0
    criterion = torch.nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            predicted = output.argmax(1)
            correct += (predicted == y).sum().item()
            cur_examples = x.size(0)
            total_loss += (loss.item() * cur_examples) 
            total_examples += cur_examples
    cur_loss = total_loss / total_examples
    cur_acc  = correct / total_examples
    return cur_loss, cur_acc

def save_model(model, opt, config):
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

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_path', type=str, default='data/snips/embedding.txt')
    parser.add_argument('--label_path', type=str, default='data/snips/label.txt')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--save_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--l2norm', type=float, default=1e-6)
    parser.add_argument('--tmax',type=int, default=-1)
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument('--log_dir', type=str, default='runs')

    opt = parser.parse_args()

    # training default device : GPU
    device = torch.device("cuda")

    # random seed
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)

    # APEX and distributed setting
    if not APEX_AVAILABLE: opt.use_amp = False
    opt.distributed = False
    if 'WORLD_SIZE' in os.environ and opt.use_amp:
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.distributed = int(os.environ['WORLD_SIZE']) > 1
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.word_size = torch.distributed.get_world_size()

    try:
        with open(opt.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        config = dict()
   
    # prepare train, valid dataset
    opt.train_data_path = os.path.join(opt.data_dir, 'train.txt.ids')
    train_dataset = SnipsDataset(opt.train_data_path)
    train_sampler = None
    if opt.distributed:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, \
            shuffle=False, num_workers=2, sampler=train_sampler)
    print("[Train data loaded]")
    opt.valid_data_path = os.path.join(opt.data_dir, 'valid.txt.ids')
    valid_dataset = SnipsDataset(opt.valid_data_path)
    valid_sampler = None
    if opt.distributed:
        valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, \
            shuffle=False, num_workers=2, sampler=valid_sampler)
    print("[Valid data loaded]")

    # create model, optimizer, scheduler, summary writer
    print("[Creating Model, optimizer, scheduler, summary writer...]")
    model = TextCNN(config, opt.embedding_path, opt.label_path)
    model.to(device)
    opt.one_epoch_step = (len(train_dataset) // (opt.batch_size*opt.world_size))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2norm)
    if opt.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.opt_level)
    if opt.distributed:
        model = DDP(model, delay_allreduce=True)
    scheduler = None
    try:
        writer = SummaryWriter(log_dir=opt.log_dir)
    except:
        writer = None
    print (opt)
    print("[Ready]")

    # training
    config['device'] = device
    config['optimizer'] = optimizer
    config['scheduler'] = scheduler
    config['writer'] = writer
    config['opt'] = opt
    best_val_loss = float('inf')
    for epoch_i in range(opt.epoch):
        epoch_st_time = time.time()
        eval_loss = train_epoch(model, config, train_loader, valid_loader, epoch_i+1)
        if opt.local_rank == 0 and eval_loss < best_val_loss:
            best_val_loss = eval_loss
            if opt.save_path:
                save_model(model, opt, config)   
   
if __name__ == '__main__':
    main()
