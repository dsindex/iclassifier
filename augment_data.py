"""
code from https://github.com/tacchinotacchi/distil-bilstm/blob/master/generate_dataset.py
"""

import os
import argparse
import numpy as np
from tqdm.autonotebook import tqdm
import csv
import spacy
from spacy.symbols import ORTH
spacy_en = spacy.load('en_core_web_sm')
mask_token = '<mask>'
spacy_en.tokenizer.add_special_case(mask_token, [{ORTH: mask_token}])

def load_tsv(path, skip_header=True):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        if skip_header:
            next(reader)
        data = [row for row in reader]
    return data

def build_pos_dict(sentences):
    pos_dict = {}
    for sentence in sentences:
        for word in sentence:   
            pos_tag = word.pos_
            if pos_tag not in pos_dict:
                pos_dict[pos_tag] = []
            if word.text.lower() not in pos_dict[pos_tag]:
                pos_dict[pos_tag].append(word.text.lower())
    return pos_dict

def make_sample(input_sentence, pos_dict, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
    sentence = []
    for word in input_sentence:
        # Apply single token masking or POS-guided replacement
        u = np.random.uniform()
        if u < p_mask:
            sentence.append(mask_token)
        elif u < (p_mask + p_pos):
            same_pos = pos_dict[word.pos_]
            # Pick from list of words with same POS tag
            sentence.append(np.random.choice(same_pos))
        else:
            sentence.append(word.text.lower())
    # Apply n-gram sampling
    if len(sentence) > 2 and np.random.uniform() < p_ng:
        n = min(np.random.choice(range(1, 5+1)), len(sentence) - 1)
        start = np.random.choice(len(sentence) - n)
        for idx in range(start, start + n):
            sentence[idx] = mask_token
    return sentence

def augmentation(sentences, pos_dict, n_iter=20, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
    augmented = []
    for sentence in tqdm(sentences, 'Generation'):
        samples = [[word.text.lower() for word in sentence]]
        for _ in range(n_iter):
            new_sample = make_sample(sentence, pos_dict, p_mask, p_pos, p_ng, max_ng)
            if new_sample not in samples:
                samples.append(new_sample)
        augmented.extend(samples)
    return augmented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input dataset.")
    parser.add_argument('--output', type=str, required=True, help="Output dataset.")
    parser.add_argument('--lang', type=str, default='en')
    args = parser.parse_args()
    
    # Load original tsv file
    input_tsv = load_tsv(args.input, skip_header=False)

    if args.lang == 'en':
        sentences = [spacy_en(text) for text, _ in tqdm(input_tsv, desc='Loading dataset')]
    # build lists of words indexes by POS tab
    pos_dict = build_pos_dict(sentences)

    # Generate augmented samples
    sentences = augmentation(sentences, pos_dict)

    # Write to file
    with open(args.output, 'w') as f:
        for sentence in sentences:
            f.write("%s\n" % ' '.join(sentence))
