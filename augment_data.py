"""
base code from https://github.com/tacchinotacchi/distil-bilstm/blob/master/generate_dataset.py
"""

import os
import argparse
import numpy as np
from tqdm.autonotebook import tqdm
import csv
import multiprocessing as mp
from functools import reduce

class Word:
    def __init__(self, word, pos):
        self.text = word
        self.pos_ = pos

    def __str__(self):
        return '{}/{}'.format(self.text, self.pos_)

def load_tsv(path, skip_header=True):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        if skip_header:
            next(reader)
        data = [row for row in reader]
    return data

def build_pos_dict(sentences, lower=True):
    pos_dict = {}
    for sentence in sentences:
        for word in sentence:   
            pos_tag = word.pos_
            if pos_tag not in pos_dict:
                pos_dict[pos_tag] = []
            w = word.text
            if lower: w = w.lower()
            if w not in pos_dict[pos_tag]:
                pos_dict[pos_tag].append(w)
    return pos_dict

def make_sample(input_sentence, pos_dict, lower=True):
    # fixed hyperparams for sampling
    p_mask = 0.1 # mask prob
    p_pos = 0.1  # pos prob
    p_ng = 0.25  # ngram prob
    max_ng = 5   # ngram size

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
            w = word.text
            if lower: w = w.lower()
            sentence.append(w)
    # Apply n-gram sampling
    if len(sentence) > 2 and np.random.uniform() < p_ng:
        n = min(np.random.choice(range(1, max_ng+1)), len(sentence) - 1)
        start = np.random.choice(len(sentence) - n)
        for idx in range(start, start + n):
            sentence[idx] = mask_token
    return sentence

def make_samples(entry):
    sentence = entry['sentence']
    pos_dict = entry['pos_dict']
    lower = entry['lower']
    n_iter = entry['n_iter']
    samples = [[word.text.lower() if lower else word.text for word in sentence]]
    for _ in range(n_iter):
        new_sample = make_sample(sentence, pos_dict, lower)
        if new_sample not in samples:
            samples.append(new_sample)
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input dataset.")
    parser.add_argument('--output', type=str, required=True, help="Output dataset.")
    parser.add_argument('--mask_token', type=str, default='[MASK]')
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--dummy_label', type=str, default='dummy')
    parser.add_argument('--lang', type=str, default='en', help="Target language, 'en'|'ko', default 'en'.")
    parser.add_argument('--lower', action='store_true', help="Enable lowercase.")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing for sampling.")
    args = parser.parse_args()
    
    # Load original tsv file
    input_tsv = load_tsv(args.input, skip_header=False)

    # POS tagging
    mask_token = args.mask_token
    if args.lang == 'en':
        import spacy
        from spacy.symbols import ORTH
        spacy_en = spacy.load('en_core_web_sm')
        spacy_en.tokenizer.add_special_case(mask_token, [{ORTH: mask_token}])
        sentences = [spacy_en(text) for text, _ in tqdm(input_tsv, desc='POS tagging')]
    if args.lang == 'ko':
        from khaiii import KhaiiiApi
        khaiii_api = KhaiiiApi()
        sentences = []
        for text, _ in tqdm(input_tsv, desc='POS tagging'):
            sentence = []
            khaiii_sentence = khaiii_api.analyze(text)
            for khaiii_word in khaiii_sentence:
                tags = []
                for khaiii_morph in khaiii_word.morphs:
                    morph = khaiii_morph.lex
                    tag = khaiii_morph.tag
                    # add '-다' for matching GloVe vocab.
                    if tag in ['VV', 'VA', 'VX', 'XSV', 'XSA', 'VCP']: morph += u'다'
                    word = Word(morph, tag)
                    sentence.append(word)
            sentences.append(sentence) 

    # Build lists of words indexes by POS
    pos_dict = build_pos_dict(sentences, lower=args.lower)

    # Generate augmented samples
    if args.parallel:
        pool = mp.Pool(mp.cpu_count())
        # processs in parallel
        entries = []
        for sentence in tqdm(sentences, desc='Preparation data for multiprocessing'):
            entry = {'sentence': sentence, 'pos_dict': pos_dict, 'lower': args.lower, 'n_iter': args.n_iter}
            entries.append(entry)
        print('Data ready! go parallel!') 
        sentences = pool.map(make_samples, entries, chunksize=100)
        sentences = reduce(lambda x,y: x+y, sentences)
        pool.close()
        pool.join()
        print('Done!')
    else:
        # process sequentially
        augmented = []
        for sentence in tqdm(sentences, desc='Sampling'):
            entry = {'sentence': sentence, 'pos_dict': pos_dict, 'lower': args.lower, 'n_iter': args.n_iter}
            samples = make_samples(entry) 
            augmented.extend(samples)
        sentences = augmented

    # Write to file
    with open(args.output, 'w') as f:
        for sentence in tqdm(sentences, desc='Writing'):
            f.write("{}\t{}\n".format(' '.join(sentence), args.dummy_label))
