from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextGloveCNN(nn.Module):
    def __init__(self, config, embedding_path, label_path, emb_non_trainable=False):
        super(TextGloveCNN, self).__init__()

        token_emb_dim = config['token_emb_dim']
        seq_size = config['n_ctx']
        kernel_sizes = config['kernel_sizes']
        num_filters = config['num_filters']

        # 1. glove embeddig
        weights_matrix = self.__load_embedding(embedding_path)
        self.embed = self.__create_embedding_layer(weights_matrix, non_trainable=emb_non_trainable)

        # 2. convolution
        convs = []
        for ks in kernel_sizes:
            convs.append(nn.Conv1d(in_channels=token_emb_dim, out_channels=num_filters, kernel_size=ks))
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(config['dropout'])

        # 3. fully connected
        self.labels = self.__load_label(label_path)
        label_size = len(self.labels)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, label_size)

    def __load_embedding(self, input_path):
        weights_matrix = np.load(input_path)
        weights_matrix = torch.tensor(weights_matrix)
        return weights_matrix

    def __create_embedding_layer(self, weights_matrix, non_trainable=False):
        vocab_size, emb_dim = weights_matrix.size()
        emb_layer = nn.Embedding(vocab_size, emb_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    def __load_label(self, input_path):
        labels = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                label = toks[0]
                label_id = toks[1]
                labels[label_id] = label
        return labels

    def forward(self, x):
        # 1. glove embedding
        # [batch_size, seq_size]
        embedded = self.embed(x)
        # [batch_size, seq_size, token_emb_dim]  

        # 2. convolution
        embedded = embedded.permute(0, 2, 1)
        # [batch_size, token_emb_dim, seq_size]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # [ [batch_size, num_filters, *], [batch_size, num_filters, *], [batch_size, num_filters, *] ]
        
        pooled = [F.max_pool1d(conv, int(conv.size(2))).squeeze(2) for conv in conved]
        # [ [batch_size, num_filters], [batch_size, num_filters], [batch_size, num_filters] ]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        # [batch_size, len(kernel_sizes) * num_filters]

        # 3. fully connected
        output = torch.sigmoid(self.fc(cat))
        # [batch_size, label_size]
        return output

class TextBertCNN(nn.Module):
    def __init__(self, config, bert_config, bert_model, label_path, feature_based=False):
        super(TextBertCNN, self).__init__()

        seq_size = config['n_ctx']
        kernel_sizes = config['kernel_sizes']
        num_filters = config['num_filters']

        # 1. bert embedding
        self.bert_config = bert_config
        self.bert_model = bert_model
        hidden_size = bert_config.hidden_size
        self.feature_based = feature_based

        # 2. convolution
        convs = []
        for ks in kernel_sizes:
            convs.append(nn.Conv1d(in_channels=hidden_size, out_channels=num_filters, kernel_size=ks))
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(config['dropout'])

        # 3. fully connected
        self.labels = self.__load_label(label_path)
        label_size = len(self.labels)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, label_size)

    def __load_label(self, input_path):
        labels = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                label = toks[0]
                label_id = toks[1]
                labels[label_id] = label
        return labels

    def __compute_bert_embedding(self, x):
        if self.feature_based:
            # feature-based
            with torch.no_grad():
                bert_outputs = self.bert_model(input_ids=x[0],
                                               attention_mask=x[1],
                                               token_type_ids=x[2])
                embedded = bert_outputs[0]
        else:
            # fine-tuning
            # x[0], x[1], x[2] : [batch_size, seq_size]
            bert_outputs = self.bert_model(input_ids=x[0],
                                           attention_mask=x[1],
                                           token_type_ids=x[2])
            embedded = bert_outputs[0]
            # [batch_size, seq_size, hidden_size]
            # [batch_size, seq_size, 0] corresponding to [CLS]
        return embedded

    def forward(self, x):
        # 1. bert embedding
        embedded = self.__compute_bert_embedding(x)
        embedded = self.dropout(embedded)

        # 2. convolution
        embedded = embedded.permute(0, 2, 1)
        # [batch_size, hidden_size, seq_size]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # [ [batch_size, num_filters, *], [batch_size, num_filters, *], [batch_size, num_filters, *] ]
        
        pooled = [F.max_pool1d(conv, int(conv.size(2))).squeeze(2) for conv in conved]
        # [ [batch_size, num_filters], [batch_size, num_filters], [batch_size, num_filters] ]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        # [batch_size, len(kernel_sizes) * num_filters]

        # 3. fully connected
        output = torch.sigmoid(self.fc(cat))
        # [batch_size, label_size]
        return output
