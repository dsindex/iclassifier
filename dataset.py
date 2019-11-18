from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset

class SnipsGLOVEDataset(Dataset):
    def __init__(self, path):
        x,y = [],[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                items = line.strip().split()
                yi = float(items[0])
                xi = [int(d) for d in items[1:]]
                x.append(xi)
                y.append(yi)
        
        self.x = torch.tensor(x) 
        self.y = torch.tensor(y).long()
 
    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SnipsBERTDataset(Dataset):
    def __init__(self, path):
        features = torch.load(path)
        # convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

        self.dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id)
 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
