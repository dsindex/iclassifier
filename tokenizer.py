from __future__ import absolute_import, division, print_function

import os
import pdb

class Tokenizer():
    def __init__(self, config):
        self.config = config

    def tokenize(self, sent):
        # white space tokenizer
        tokens = sent.split()
        return tokens
