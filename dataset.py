from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
from torch.utils.data.dataset import Dataset

class SnipsDataset(Dataset):
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

