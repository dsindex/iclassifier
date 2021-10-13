import sys
import os
import argparse
import random
import json
from collections import Counter 
import pdb
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def proc(input_path):
    tot_num_line = sum(1 for _ in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            if idx == 0: continue
            toks = line.strip().split('\t')
            sent_a = toks[8]
            sent_b = toks[9]
            label = toks[-1]
            print(f"{sent_a}\t{sent_b}\t{label}")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, default='train.tsv')
    opt = parser.parse_args()

    proc(opt.input_path)


if __name__ == '__main__':
    main()
