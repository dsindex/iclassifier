from abc import ABC
import json
import logging
import os

import torch
import torch.quantization
import numpy as np
from model import TextGloveCNN, TextGloveDensenetCNN, TextGloveDensenetDSA, TextBertCNN, TextBertCLS
from util import load_config, to_device, to_numpy

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class Opt:
    def __init__(self, model_dir):
        self.config = os.path.join(model_dir, 'mconfig.json')
        self.model_path = os.path.join(model_dir, 'pytorch-model.pt')
        self.device = 'cpu'
        self.num_threads = 14
        self.enable_dqm = True
        self.label_path = os.path.join(model_dir, 'label.txt')
        # for emb_class='glove'
        self.embedding_path = os.path.join(model_dir, 'embedding.npy')
        self.vocab_path = os.path.join(model_dir, 'mvocab.txt')
        # for emb_class='bert'
        self.bert_output_dir = model_dir

class ClassifierHandler(BaseHandler, ABC):

    def __init__(self):
        super(ClassifierHandler, self).__init__()
        self.initialized = False

    def load_checkpoint(self):
        config = self.config
        opt = config['opt']
        if opt.device == 'cpu':
            checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(opt.model_path)
        logger.info("[Loading checkpoint done] %s" % (opt.model_path))
        return checkpoint

    def load_model(self, checkpoint):
        config = self.config
        opt = config['opt']
        if config['emb_class'] == 'glove':
            if config['enc_class'] == 'cnn':
                model = TextGloveCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
            if config['enc_class'] == 'densenet-cnn':
                model = TextGloveDensenetCNN(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
            if config['enc_class'] == 'densenet-dsa':
                model = TextGloveDensenetDSA(config, opt.embedding_path, opt.label_path, emb_non_trainable=True)
        if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
            from transformers import BertTokenizer, BertConfig, BertModel
            from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel
            from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
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
            Config    = MODEL_CLASSES[config['emb_class']][0]
            Tokenizer = MODEL_CLASSES[config['emb_class']][1]
            Model     = MODEL_CLASSES[config['emb_class']][2]
            bert_config = Config.from_pretrained(opt.bert_output_dir)
            bert_tokenizer = Tokenizer.from_pretrained(opt.bert_output_dir)
            # no need to use 'from_pretrained'
            bert_model = Model(bert_config)
            ModelClass = TextBertCNN
            if config['enc_class'] == 'cls': ModelClass = TextBertCLS
            model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path)
        model.load_state_dict(checkpoint)
        model = model.to(opt.device)
        logger.info("[Model loaded]")
        return model

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = {}
            for idx, line in enumerate(f):
                tokens = line.split()
                word = tokens[0]
                word_id = int(tokens[1])
                vocab[word] = word_id
            return vocab

    def prepare_tokenizer(self):
        from tokenizer import Tokenizer
        config = self.config
        opt = config['opt']
        model = self.model
        if config['emb_class'] == 'glove':
            vocab = self.load_vocab(opt.vocab_path)
            tokenizer = Tokenizer(vocab, config)
        if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
            tokenizer = model.bert_tokenizer
        return tokenizer

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        # set config, path
        properties = ctx.system_properties
        model_dir = properties.get('model_dir')
        logger.info("model_dir: %s", model_dir)
        opt = Opt(model_dir)
        with open(opt.config, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        logger.info("%s", self.config)
        if opt.num_threads > 0: torch.set_num_threads(opt.num_threads)
        self.config['opt'] = opt
        logger.info("opt.device: %s", opt.device)
        logger.info("opt.num_threads: %s", opt.num_threads)
        logger.info("opt.enable_dqm: %s", opt.enable_dqm)

        # load pytorch model checkpoint
        checkpoint = self.load_checkpoint()
    
        # prepare model and load parameters
        self.model = self.load_model(checkpoint)
        self.model.eval()

        # enable to use dynamic quantized model (pytorch>=1.3.0)
        if opt.enable_dqm and opt.device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            print(self.model)

        # prepare tokenizer
        self.tokenizer = self.prepare_tokenizer()

        # prepare labels
        self.labels = self.model.labels

        self.initialized = True

    def encode_text(self, text):
        config = self.config
        tokenizer = self.tokenizer
        if config['emb_class'] == 'glove':
            tokens = text.split()
            # kernel size can't be greater than actual input size,
            # we should pad the sequence up to the maximum kernel size + 1.
            min_seq_size = 10
            ids = tokenizer.convert_tokens_to_ids(tokens, pad_sequence=False, min_seq_size=min_seq_size)
            x = torch.tensor([ids])
            # x : [batch_size, variable size]
            # batch size: 1
        if config['emb_class'] in ['bert', 'distilbert', 'albert', 'roberta', 'bart', 'electra']:
            from torch.utils.data import TensorDataset
            inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
            if config['emb_class'] in ['bart', 'distilbert']:
                x = [inputs['input_ids'], inputs['attention_mask']]
                # x[0], x[1] : [batch_size, variable size]
            else:
                x = [inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']]
                # x[0], x[1], x[2] : [batch_size, variable size]
            # batch size: 1
        return x

    def preprocess(self, data):
        config = self.config
        opt = config['opt']
        logger.info("data: %s", data)
        text = data[0].get('data')
        if text is None:
            text = data[0].get('body')
        if text:
            text = text.decode('utf-8')
        logger.info("[Received text] %s", text)
        x = self.encode_text(text)
        x = to_device(x, opt.device)
        return x, text

    def inference(self, data):
        config = self.config
        opt = config['opt']
        model = self.model
        labels = self.labels

        logits = model(data)
        predicted = logits.argmax(1)
        predicted = to_numpy(predicted)[0]
        predicted_raw = labels[predicted]
        logger.info("[Model predicted] %s", predicted_raw)

        return predicted_raw

    def postprocess(self, data, text):
        return [
            {'text': text,
             'results': data},
        ]

_service = ClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data, text = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data, text)

        return data
    except Exception as e:
        raise e
