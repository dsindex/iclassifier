from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from transformers import BartConfig, BartTokenizer, BartModel
from transformers import ElectraConfig, ElectraTokenizer, ElectraModel
MODEL_CLASSES = {
    "bert": (BertConfig, BertTokenizer, BertModel),
    "distilbert": (DistilBertConfig, DistilBertTokenizer, DistilBertModel),
    "albert": (AlbertConfig, AlbertTokenizer, AlbertModel),
    "roberta": (RobertaConfig, RobertaTokenizer, RobertaModel),
    "bart": (BartConfig, BartTokenizer, BartModel),
    "electra": (ElectraConfig, ElectraTokenizer, ElectraModel),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_type", type=str, default='bert',
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-cased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-cased)")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    opt = parser.parse_args()

    # mapping config, tokenizer, model class for model_type
    Config    = MODEL_CLASSES[opt.model_type][0]
    Tokenizer = MODEL_CLASSES[opt.model_type][1]
    Model     = MODEL_CLASSES[opt.model_type][2]

    # download
    logger.info("[Downloading transformers...]")
    tokenizer = Tokenizer.from_pretrained(opt.model_name_or_path,
                                          do_lower_case=opt.do_lower_case)
    model = Model.from_pretrained(opt.model_name_or_path,
                                  from_tf=bool(".ckpt" in opt.model_name_or_path))
    config = model.config
    logger.info("[Done]")
    # save
    output_dir = opt.model_name_or_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    logger.info("[Saved to {}]".format(output_dir))
 
if __name__ == '__main__':
    main()
