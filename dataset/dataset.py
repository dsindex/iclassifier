import os
import pdb

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(config, filepath, DatasetClass, sampling=False, num_workers=1, batch_size=0, hp_search_bsz=None):
    args = config['args']
    dataset = DatasetClass(config, filepath)

    if sampling:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    bz = args.batch_size
    if batch_size > 0: bz = batch_size
    # for optuna
    if hp_search_bsz: bz = hp_search_bsz

    loader = DataLoader(dataset, batch_size=bz, num_workers=num_workers, sampler=sampler, pin_memory=True)
    logger.info("[{} data loaded]".format(filepath))
    return loader

class GloveDataset(Dataset):
    def __init__(self, config, path):
        x,y = [],[]
        logits_as_label = False
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                y_data, x_data = line.strip().split('\t')
                if len(y_data.split()) >= 2: # soft label
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
    def __init__(self, config, path):
        dataset = torch.load(path)

        all_input_ids = torch.tensor([f for f in dataset['input_ids']], dtype=torch.long)
        all_attention_mask = torch.tensor([f for f in dataset['attention_mask']], dtype=torch.long)
        probe_label_id = dataset['label'][0]

        if config['emb_class'] in ['roberta', 'bart', 'distilbert', 'ibert', 't5']:
            self.x = TensorDataset(all_input_ids, all_attention_mask)
        else:
            all_token_type_ids = torch.tensor([f for f in dataset['token_type_ids']], dtype=torch.long)
            self.x = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids) 

        if len(str(probe_label_id).split()) >= 2: # soft label
            all_label = torch.tensor([[float(logit) for logit in str(f).split()] for f in dataset['label']])
        else:
            all_label = torch.tensor([f for f in dataset['label']], dtype=torch.long)

        self.y = all_label
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

