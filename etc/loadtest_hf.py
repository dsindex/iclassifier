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
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    opt = parser.parse_args()

    # load
    logger.info("[Loading...]")
    tokenizer = AutoTokenizer.from_pretrained(opt.model_path,
                                          do_lower_case=opt.do_lower_case)
    model = AutoModel.from_pretrained(opt.model_path,
                                  from_tf=bool(".ckpt" in opt.model_path))
    config = model.config
    logger.info("[Done]")
    # save
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    tokenizer.save_pretrained(opt.output_dir)
    model.save_pretrained(opt.output_dir)
    logger.info("[Saved to {}]".format(opt.output_dir))
 
if __name__ == '__main__':
    main()
