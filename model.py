import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextCNN(nn.Module):
    def __init__(self, config, embedding_path, label_path):
        super(TextCNN, self).__init__()

        vocab_size = config['vocab_size']
        token_emb_dim = config['token_emb_dim']
        seq_size = config['n_ctx']
        kernel_sizes = config['kernel_sizes']
        num_filters = config['num_filters']

        # embeddig
        weights_matrix = self.load_embedding(embedding_path, vocab_size, token_emb_dim)
        self.embed = self.create_embedding_layer(weights_matrix, non_trainable=False)

        # convolution
        convs = []
        for ks in kernel_sizes:
            convs.append(nn.Conv1d(in_channels=token_emb_dim, output_channels=num_filters, kernel_size=ks))
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(config['dropout'])

        # fully connected
        self.labels = self.load_label(label_path)
        self.label_size = len(self.labels)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, label_size)

    def load_embedding(input_path, vocab_size, token_emb_dim):
        weights_matrix = np.zeros((vocab_size, token_emb_dim))
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                tok_id = int(toks[0])
                vector = np.array(toks[1:]).astype(np.float)
                weights_matrix[tok_id] = vector
        return weights_matrix

    def create_embedding_layer(weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    def load_label(input_path):
        labels = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                label = toks[0]
                label_id = toks[1]
                labels[label_id] = label
        return labels

    def forward(self, x):
        # [batch_size, seq_size]
        embedded = self.embed(x)
        # [batch_size, seq_size, token_emb_dim]  

        embedded = embedded.permute(0, 2, 1)
        # [batch_size, token_emb_dim, seq_size]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # [ [batch_size, num_filters, *], [batch_size, num_filters, *], [batch_size, num_filters, *] ]
        
        pooled = [F.max_pool1d(conv, int(conv.size(2))).squeeze(2) for conv in conved]
        # [ [batch_size, num_filters], [batch_size, num_filters], [batch_size, num_filters] ]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        # [batch_size, len(kernel_sizes) * num_filters]

        output = torch.sigmoid(self.fc(cat))
        # [batch_size, label_size]
        return output


