from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
from collections import Counter 
import pdb
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(data_path):
    logger.info("[read data]")
    data = []
    tot_num_line = sum(1 for _ in open(data_path, 'r')) 
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            data.append(line)
    logger.info("number of lines : {}".format(len(data)))
    return data

def write_data(data, base_path):
    logger.info("[write data]")

    train_path = base_path + '.train'
    valid_path = base_path + '.valid'
    test_path = base_path + '.test'
    tot_num_docs = len(data)
    limit = tot_num_docs // 10
    f_train = open(train_path, 'w', encoding='utf-8')
    f_valid = open(valid_path, 'w', encoding='utf-8')
    f_test = open(test_path, 'w', encoding='utf-8')

    random.shuffle(data)

    for idx, line in enumerate(tqdm(data)):
        if idx < limit: # 10%
            f_valid.write(line + '\n')
        elif idx >= limit and idx < limit*3: # 20%
            f_test.write(line + '\n')
        else: # 70%
            f_train.write(line + '\n')
    f_train.close()
    f_valid.close()
    f_test.close()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='total.txt')
    parser.add_argument('--base_path', type=str, default='data')
    opt = parser.parse_args()

    data = read_data(opt.data_path)
    write_data(data, opt.base_path)

if __name__ == '__main__':
    main()
