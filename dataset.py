from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(config, filepath, DatasetClass, sampling=False, num_workers=1, batch_size=0, hp_search_bsz=None):
    opt = config['opt']
    dataset = DatasetClass(filepath)

    if sampling:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    if hasattr(opt, 'distributed') and opt.distributed:
        sampler = DistributedSampler(dataset)

    bz = opt.batch_size
    if batch_size > 0: bz = batch_size
    # for optuna
    if hp_search_bsz: bz = hp_search_bsz

    loader = DataLoader(dataset, batch_size=bz, num_workers=num_workers, sampler=sampler, pin_memory=True)
    logger.info("[{} data loaded]".format(filepath))
    return loader

class GloveDataset(Dataset):
    def __init__(self, path):
        x,y = [],[]
        logits_as_label = False
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                y_data, x_data = line.strip().split('\t')
                if len(y_data.split()) >= 2: # logits as label
                    yi = [float(f) for f in y_data.split()]
                    if logits_as_label is False: logits_as_label = True
                else:
                    yi = float(y_data)
                xi = [int(d) for d in x_data.split()]
                x.append(xi)
                y.append(yi)

        self.x = torch.tensor(x)
        if logits_as_label:
            self.y = torch.tensor(y)
        else:
            self.y = torch.tensor(y).long()
 
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class BertDataset(Dataset):
    def __init__(self, path):
        # load features from file
        features = torch.load(path)
        # convert to tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        probe_label_id = features[0].label_id
        if len(str(probe_label_id).split()) >= 2: # logits as label
            all_label_id = torch.tensor([[float(logit) for logit in str(f.label_id).split()] for f in features])
        else:
            all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

        self.x = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        self.y = all_label_id
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
