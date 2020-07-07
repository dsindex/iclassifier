from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import BertTokenizer, BertConfig, BertModel, BertForMaskedLM
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from transformers import ElectraConfig, ElectraTokenizer, ElectraModel, ElectraForMaskedLM
MODEL_CLASSES = {
    "bert": (BertConfig, BertTokenizer, BertModel, BertForMaskedLM),
    "roberta": (RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM),
    "electra": (ElectraConfig, ElectraTokenizer, ElectraModel, ElectraForMaskedLM),
}

from transformers.modeling_longformer import LongformerSelfAttention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_long_model(model_type, model, tokenizer, config, save_model_to, attention_window=512, max_pos=4096):
    """Convert RoBERTa to Longformer.
    for other model_type like BERT, replacing model.encoder.layer.attention.self to LongformerSelfAttension()
    is not available at this time.
    """
    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape

    if model_type in ['roberta']:
        max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2

    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos

    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    b = 0
    if model_type in ['roberta']: # NOTE: RoBERTa has positions 0,1 reserved
        k = 2
        step = current_max_pos - 2
        b = 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[b:]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    # save Long version for RoBERTa only.
    # for other model_type, you can use its long version on-the-fly(maybe?).
    if model_type in ['roberta']:
        logger.info(f'saving model to {save_model_to}')
        model.save_pretrained(save_model_to)
        tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_type", type=str, default='bert',
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    opt = parser.parse_args()

    # mapping config, tokenizer, model class for model_type
    Config    = MODEL_CLASSES[opt.model_type][0]
    Tokenizer = MODEL_CLASSES[opt.model_type][1]
    Model     = MODEL_CLASSES[opt.model_type][2]

    # load pretrained model
    logger.info("[Loading...]")
    tokenizer = Tokenizer.from_pretrained(opt.model_path,
                                          do_lower_case=opt.do_lower_case)
    model = Model.from_pretrained(opt.model_path,
                                  from_tf=bool(".ckpt" in opt.model_path))
    config = model.config
    logger.info("[Done]")

    # convert to long version
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    model, tokenizer = create_long_model(opt.model_type, model, tokenizer, config, opt.output_dir, attention_window=512, max_pos=4096)
 
if __name__ == '__main__':
    main()
