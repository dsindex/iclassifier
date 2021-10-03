import os
import pdb
import json
import torch

def load_checkpoint(model_path, device='cuda'):
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=torch.device(device))
    else:
        checkpoint = torch.load(model_path)
    return checkpoint

def load_config(args, config_path=None):
    try:
        if not config_path: config_path = args.config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        config = dict()
    return config

def load_label(label_path):
    labels = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            toks = line.strip().split()
            label = toks[0]
            label_id = int(toks[1])
            labels[label_id] = label
    return labels

def to_device(x, device):
    if type(x) != list: # torch.tensor
        x = x.to(device)
    else:               # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].to(device)
    return x

def to_numpy(x):
    if type(x) != list: # torch.tensor
        x = x.detach().cpu().numpy()
    else:               # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].detach().cpu().numpy()
    return x
