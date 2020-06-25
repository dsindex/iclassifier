"""
base code from https://github.com/tacchinotacchi/distil-bilstm/blob/master/generate_dataset.py
"""
import sys
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

def make_sample(entry):
    input_sentence = entry['sentence']
    pos_dict = entry['pos_dict']
    lower = entry['args'].lower
    p_mask = entry['args'].p_mask
    p_pos = entry['args'].p_pos
    p_ng = entry['args'].p_ng
    max_ng = entry['args'].max_ng
    analyzer = entry['args'].analyzer
    mask_token = entry['args'].mask_token

    num_tokens = len(input_sentence)
    max_mask_count = num_tokens // 2 # max portion of masked tokens == 50%
    mask_count = 0
    sentence = []
    for word in input_sentence:
        u = np.random.uniform()
        if u < (p_mask + p_pos) and mask_count <= max_mask_count:
            # Apply single token masking or POS-guided replacement
            if u < p_mask:
                sentence.append(mask_token)
                mask_count += 1
            elif u < (p_mask + p_pos):
                if analyzer == 'spacy':
                    same_pos = pos_dict[word.pos_]
                    # Pick from list of words with same POS tag
                    sentence.append(np.random.choice(same_pos))
                    mask_count += 1
                else: # analyzer in ['khaiii', 'npc']
                    if word.pos_[0] in ['J', 'E'] or word.pos_ in ['VX', 'NNB']: # exclude 'Josa, Eomi', 'Auxiliary verb', 'Bound noun' 
                        w = word.text
                        if lower: w = w.lower()
                        sentence.append(w)
                    else:
                        same_pos = pos_dict[word.pos_]
                        # Pick from list of words with same POS tag
                        sentence.append(np.random.choice(same_pos))
                        mask_count += 1
        else:
            w = word.text
            if lower: w = w.lower()
            sentence.append(w)

    # Apply n-gram sampling
    if len(sentence) > 2 and np.random.uniform() < p_ng:
        n = min(np.random.choice(range(1, max_ng+1)), len(sentence) - 1)
        if mask_count + n <= max_mask_count:
            start = np.random.choice(len(sentence) - n)
            for idx in range(start, start + n):
                sentence[idx] = mask_token
            mask_count += n
    return sentence

def make_samples(entry):
    sentence = entry['sentence']
    lower = entry['args'].lower
    # hyperparams for sampling : p_mask, p_pos, p_ng, max_ng, n_iter
    n_iter = entry['args'].n_iter
    samples = [[word.text.lower() if lower else word.text for word in sentence]]
    for _ in range(n_iter):
        new_sample = make_sample(entry)
        if new_sample not in samples:
            samples.append(new_sample)
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input dataset.")
    parser.add_argument('--output', type=str, required=True, help="Output dataset.")
    parser.add_argument('--mask_token', type=str, default='[MASK]')
    parser.add_argument('--p_mask', type=float, default=0.1, help="Prob for masking single token.")
    parser.add_argument('--p_pos', type=float, default=0.1, help="Prob for replacing single token using POS.")
    parser.add_argument('--p_ng', type=float, default=0.25, help="Prob for masking ngram.")
    parser.add_argument('--max_ng', type=int, default=5, help="Max ngram size for masking.")
    parser.add_argument('--n_iter', type=int, default=20, help="Number of iteration for sampling.")
    parser.add_argument('--dummy_label', type=str, default='dummy')
    parser.add_argument('--analyzer', type=str, default='spacy', help="Analyzer, 'spacy | khaiii | npc', default 'spacy'.")
    parser.add_argument('--lower', action='store_true', help="Enable lowercase.")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing for sampling.")
    parser.add_argument('--no_augment', action='store_true', help="No augmentation used.")
    args = parser.parse_args()
    
    # Load original tsv file
    input_tsv = load_tsv(args.input, skip_header=False)

    # POS tagging
    if args.analyzer == 'spacy':
        import spacy
        from spacy.symbols import ORTH
        spacy_en = spacy.load('en_core_web_sm')
        spacy_en.tokenizer.add_special_case(args.mask_token, [{ORTH: args.mask_token}])
        sentences = [spacy_en(text) for text, _ in tqdm(input_tsv, desc='POS tagging')]
    if args.analyzer == 'khaiii':
        from khaiii import KhaiiiApi
        khaiii_api = KhaiiiApi()
        sentences = []
        for text, _ in tqdm(input_tsv, desc='POS tagging'):
            sentence = []
            khaiii_sentence = khaiii_api.analyze(text)
            for khaiii_word in khaiii_sentence:
                for khaiii_morph in khaiii_word.morphs:
                    morph = khaiii_morph.lex
                    tag = khaiii_morph.tag
                    # we might need to modify 'morph' for matching the vocab of GloVe.
                    # ex) if tag in ['VV', 'VA', 'VX', 'XSV', 'XSA', 'VCP']: morph += u'다'
                    word = Word(morph, tag)
                    sentence.append(word)
            sentences.append(sentence)
    if args.analyzer == 'npc':
        sys.path.append('data/clova_sentiments_morph/npc-install/lib')
        import libpnpc as pnpc
        res_path = 'data/clova_sentiments_morph/npc-install/res'
        npc = pnpc.Index()
        npc.init(res_path)
        sentences = []
        for text, _ in tqdm(input_tsv, desc='POS tagging'):
            sentence = []
            npc_sentence = npc.analyze(text)
            for item in npc_sentence:
                meta = item['meta']
                if meta != '[NOR]': continue
                morph = item['morph']
                tag = item['mtag']
                word = Word(morph, tag)
                sentence.append(word)
            sentences.append(sentence)

    if args.no_augment:
        # Write to file
        with open(args.output, 'w') as f:
            for sentence in tqdm(sentences, desc='Writing'):
                s = [] 
                for word in sentence:
                    s.append(word.text)
                f.write("{}\t{}\n".format(' '.join(s), args.dummy_label))
        return

    # Build lists of words indexes by POS
    pos_dict = build_pos_dict(sentences, lower=args.lower)

    # Generate augmented samples
    if args.parallel:
        pool = mp.Pool(mp.cpu_count())
        # processs in parallel
        entries = []
        for sentence in tqdm(sentences, desc='Preparation data for multiprocessing'):
            entry = {'sentence': sentence,
                     'pos_dict': pos_dict,
                     'args': args}
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
            entry = {'sentence': sentence,
                     'pos_dict': pos_dict,
                     'args': args}
            samples = make_samples(entry) 
            augmented.extend(samples)
        sentences = augmented

    # Write to file
    with open(args.output, 'w') as f:
        for sentence in tqdm(sentences, desc='Writing'):
            f.write("{}\t{}\n".format(' '.join(sentence), args.dummy_label))
