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

def prepare_dataset(config, filepath, DatasetClass, sampling=False, num_workers=1, batch_size=0, augmented=False):
    opt = config['opt']
    if augmented:
        dataset = DatasetClass(filepath, augmented=augmented)
    else:
        dataset = DatasetClass(filepath)
    if sampling:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    if hasattr(opt, 'distributed') and opt.distributed:
        sampler = DistributedSampler(dataset)
    bz = opt.batch_size
    if batch_size > 0: bz = batch_size
    loader = DataLoader(dataset, batch_size=bz, num_workers=num_workers, sampler=sampler, pin_memory=True)
    logger.info("[{} data loaded]".format(filepath))
    return loader

class SnipsGloveDataset(Dataset):
    def __init__(self, path, augmented=False):
        x,y = [],[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                y_data, x_data = line.strip().split('\t')
                if augmented:
                    yi = [float(f) for f in y_data.split()]
                else:
                    yi = float(y_data)
                xi = [int(d) for d in x_data.split()]
                x.append(xi)
                y.append(yi)

        self.x = torch.tensor(x)
        if augmented:
            self.y = torch.tensor(y)
        else:
            self.y = torch.tensor(y).long()
 
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SnipsBertDataset(Dataset):
    def __init__(self, path):
        # load features from file
        features = torch.load(path)
        # convert to tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

        self.x = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        self.y = all_label_id
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
