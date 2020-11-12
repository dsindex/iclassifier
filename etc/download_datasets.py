from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

from datasets import load_dataset, load_metric
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_name", type=str, default='glue')
    parser.add_argument("--task_name", type=str, default='sst2')
    parser.add_argument("--split", type=str, default='train')

    opt = parser.parse_args()

    # download
    logger.info("[Downloading dataset...]")
    params = [opt.dataset_name, opt.task_name]
    dataset = load_dataset(*params)

    '''
    ex)
    dataset_name = "amazon_us_reviews",  task_name = 'Video_DVD_v1_00', 'Video_v1_00'
    ex)
    customer_id
    helpful_votes
    marketplace
    product_category
    product_id
    product_parent
    product_title
    review_body
    review_date
    review_headline
    review_id
    star_rating
    total_votes
    verified_purchase
    vine
    '''

    data = dataset[opt.split]
    logger.info(data)
    count_tot_p = 0
    count_tot_n = 0
    max_length = 512
    p_score = 5
    n_score = 1
    p_label = 'positive'
    n_label = 'negative'
    for review_body, star_rating in zip(data['review_body'], data['star_rating']):
        length = len(review_body)
        if length > max_length: continue
        if star_rating >= p_score:
            count_tot_p += 1
        elif star_rating <= n_score:
            count_tot_n += 1
        else: continue
    logger.info(f"count_tot_p = {count_tot_p}, count_tot_n = {count_tot_n}")
    count_p = 0
    count_n = 0
    for review_body, star_rating in zip(data['review_body'], data['star_rating']):
        length = len(review_body)
        if length > max_length: continue
        if star_rating >= p_score:
            label = p_label
        elif star_rating <= n_score:
            label = n_label
        else: continue

        # balancing data
        if label == p_label:
            if count_p > count_tot_n*2: continue
            print(review_body + '\t' + label)
            count_p += 1
        if label == n_label:
            if count_n > count_tot_n: continue
            print(review_body + '\t' + label)
            count_n += 1
    logger.info(f"count_p = {count_p}, count_n = {count_n}")

    logger.info("[Done]")
 
if __name__ == '__main__':
    main()
