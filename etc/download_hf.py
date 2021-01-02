from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

from transformers import AutoTokenizer, AutoConfig, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-cased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-cased)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    opt = parser.parse_args()

    # download
    logger.info("[Downloading transformers...]")
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name_or_path)
    model = AutoModel.from_pretrained(opt.model_name_or_path,
                                  from_tf=bool(".ckpt" in opt.model_name_or_path))
    config = model.config
    logger.info("[Done]")
    # save
    output_dir = opt.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    logger.info("[Saved to {}]".format(output_dir))
 
if __name__ == '__main__':
    main()
