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

def proc(args):
    with open(args.input_path, 'r') as fp:
        data = json.load(fp, encoding='utf-8')
        target = data[args.dataset]
        if args.dataset == 'sst2':
            for example in target:
                sentence = example['sentence']
                label = example['label']
                print(f"{sentence}\t{label}")
        if args.dataset == 'mnli':
            for example in target:
                sentence_a = example['premise']
                sentence_b = example['hypothesis']
                label = example['label']
                print(f"{sentence_a}\t{sentence_b}\t{label}")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, default='dev.json')
    parser.add_argument('--dataset', type=str, default='sst2')
    args = parser.parse_args()

    proc(args)


if __name__ == '__main__':
    main()
