from __future__ import absolute_import, division, print_function

import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class BaseModel(nn.Module):
    def __init__(self, config=None):
        super(BaseModel, self).__init__()
        if config:
            self.set_seed(config['opt'])

    def set_seed(self, opt):
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)

    def load_embedding(self, input_path):
        weights_matrix = np.load(input_path)
        weights_matrix = torch.tensor(weights_matrix)
        return weights_matrix

    def create_embedding_layer(self, weights_matrix, non_trainable=False):
        vocab_size, emb_dim = weights_matrix.size()
        emb_layer = nn.Embedding(vocab_size, emb_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    def load_label(self, input_path):
        labels = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                toks = line.strip().split()
                label = toks[0]
                label_id = int(toks[1])
                labels[label_id] = label
        return labels

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(TextCNN, self).__init__()
        convs = []
        for ks in kernel_sizes:
            convs.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks))
            '''
            # depthwise convolution, 'out_channels' should be 'K * in_channels'
            # see https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d , https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
            convs.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, groups=in_channels))
            '''
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        # x : [batch_size, seq_size, emb_dim]
        # num_filters == out_channels
        x = x.permute(0, 2, 1)
        # x : [batch_size, emb_dim, seq_size]
        conved = [F.relu(conv(x)) for conv in self.convs]
        # conved : [ [batch_size, num_filters, *], [batch_size, num_filters, *], [batch_size, num_filters, *] ]
        pooled = [F.max_pool1d(conv, int(conv.size(2))).squeeze(2) for conv in conved]
        # pooled : [ [batch_size, num_filters], [batch_size, num_filters], [batch_size, num_filters] ]
        cat = torch.cat(pooled, dim = 1)
        # cat : [batch_size, len(kernel_sizes) * num_filters]
        return cat

class DenseNet(nn.Module):
    def __init__(self, densenet_depth, densenet_width, emb_dim, first_num_filters, num_filters, last_num_filters):
        super(DenseNet, self).__init__()
        self.densenet_depth = densenet_depth
        self.densenet_width = densenet_width
        self.densenet_block = []
        for i, ks in enumerate(self.densenet_depth):
            if i == 0:
                in_channels = emb_dim
                out_channels = first_num_filters
            else:
                in_channels = first_num_filters + num_filters * (i-1)
                out_channels = num_filters
            padding = (ks - 1)//2
            convs = []
            for _ in range(self.densenet_width):
                conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=padding)
                convs.append(conv)
            convs = nn.ModuleList(convs)
            self.densenet_block.append(convs)
        self.densenet_block = nn.ModuleList(self.densenet_block)
        ks = 1
        in_channels = emb_dim + num_filters * self.densenet_width
        out_channels = last_num_filters
        padding = (ks - 1)//2
        self.conv_last = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks, padding=padding)

    def forward(self, x, mask):
        # x     : [batch_size, seq_size, emb_dim]
        # mask  : [batch_size, seq_size]
        x = x.permute(0, 2, 1)
        # x     : [batch_size, emb_dim, seq_size]
        masks = mask.unsqueeze(-1).to(torch.float)
        # masks : [batch_size, seq_size, 1]
        masks = masks.permute(0, 2, 1)
        # masks : [batch_size, 1, seq_size]

        merge_list = []
        for j in range(self.densenet_width):
            conv_results = []
            for i, ks in enumerate(self.densenet_depth):
                if i == 0: conv_in = x
                else: conv_in  = torch.cat(conv_results, dim=-2)
                conv_out = self.densenet_block[i][j](conv_in)
                # conv_out first : [batch_size, first_num_filters, seq_size]
                # conv_out other : [batch_size, num_filters, seq_size]
                conv_out *= masks # masking, auto broadcasting along with second dimension
                conv_out = F.relu(conv_out)
                conv_results.append(conv_out)
            merge_list.append(conv_results[-1]) # last one only 
        conv_last = self.conv_last(torch.cat([x] + merge_list, dim=-2))
        conv_last *= masks
        conv_last = F.relu(conv_last)
        # conv_last : [batch_size, last_num_filters, seq_size]
        conv_last = conv_last.permute(0, 2, 1)
        # conv_last : [batch_size, seq_size, last_num_filters]
        return conv_last

class DSA(nn.Module):
    def __init__(self, config, dsa_num_attentions, dsa_input_dim, dsa_dim, dsa_r=3):
        super(DSA, self).__init__()
        self.config = config
        dsa = []
        for i in range(dsa_num_attentions):
            dsa.append(nn.Linear(dsa_input_dim, dsa_dim))
        self.dsa = nn.ModuleList(dsa)
        self.dsa_r = dsa_r # r iterations

    def __self_attention(self, x, mask, r=3):
        # x    : [batch_size, seq_size, dsa_dim]
        # mask : [batch_size, seq_size]
        # r    : r iterations
        device = self.config['device']
        # initialize
        mask = mask.to(torch.float)
        q = torch.zeros(mask.shape[0], mask.shape[-1], requires_grad=False).to(torch.float).to(device)
        # q : [batch_size, seq_size]
        z_list = []
        # iterative computing attention
        for idx in range(r):
            # attention weights
            a = torch.softmax(q.detach().clone(), dim=-1) # preventing from unreachable variable at gradient computation. 
            # a : [batch_size, seq_size]
            a *= mask
            a = a.unsqueeze(-1)
            # a : [batch_size, seq_size, 1]
            # element-wise multiplication and summation along 1 dim
            s = (a * x).sum(1)
            # s : [batch_size, dsa_dim]
            z = torch.tanh(s)
            # z : [batch_size, dsa_dim]
            z_list.append(z)
            # update q
            m = z.unsqueeze(-1)
            # m : [batch_size, dsa_dim, 1]
            q += torch.matmul(x, m).squeeze(-1)
            # q : [batch_size, seq_size]
        return z_list[-1]

    def forward(self, x, mask):
        # x     : [batch_size, seq_size, dsa_input_dim]
        # mak   : [batch_size, seq_size]
        z_list = []
        for p in self.dsa: # dsa_num_attentions
            # projection to dsa_dim
            p_out = F.leaky_relu(p(x))
            # p_out : [batch_size, seq_size, dsa_dim]
            z_j = self.__self_attention(p_out, mask, r=self.dsa_r)
            # z_j : [batch_size, dsa_dim]
            z_list.append(z_j)
        z = torch.cat(z_list, dim=-1)
        # z : [batch_size, dsa_num_attentions * dsa_dim]
        return z

class TextGloveCNN(BaseModel):
    def __init__(self, config, embedding_path, label_path, emb_non_trainable=True):
        super().__init__(config=config)

        self.config = config
        seq_size = config['n_ctx']
        token_emb_dim = config['token_emb_dim']
        num_filters = config['num_filters']
        kernel_sizes = config['kernel_sizes']
        fc_hidden_size = config['fc_hidden_size']

        # glove embedding layer
        weights_matrix = super().load_embedding(embedding_path)
        self.embed = super().create_embedding_layer(weights_matrix, non_trainable=emb_non_trainable)
        emb_dim = token_emb_dim 

        # convolution layer
        self.textcnn = TextCNN(emb_dim, num_filters, kernel_sizes)

        self.dropout = nn.Dropout(config['dropout'])

        # fully connected layer
        self.labels = super().load_label(label_path)
        label_size = len(self.labels)
        self.layernorm1 = nn.LayerNorm(len(kernel_sizes) * num_filters)
        self.fc1 = nn.Linear(len(kernel_sizes) * num_filters, fc_hidden_size)
        self.layernorm2 = nn.LayerNorm(fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, label_size)

    def forward(self, x):
        # 1. glove embedding
        # x : [batch_size, seq_size]
        embedded = self.dropout(self.embed(x))
        # embedded : [batch_size, seq_size, emb_dim]

        # 2. convolution
        textcnn_out = self.textcnn(embedded)
        # textcnn_out : [batch_size, len(kernel_sizes) * num_filters]
        textcnn_out = self.layernorm1(textcnn_out)
        textcnn_out = self.dropout(textcnn_out)

        # 3. fully connected
        fc_hidden = self.fc1(textcnn_out)
        # fc_hidden : [batch_size, fc_hidden_size]
        fc_hidden = self.layernorm2(fc_hidden)
        fc_hidden = self.dropout(fc_hidden)
        fc_out = self.fc2(fc_hidden)
        # fc_out : [batch_size, label_size]
        output = torch.softmax(fc_out, dim=-1)
        return output

class TextGloveDensenetCNN(BaseModel):
    def __init__(self, config, embedding_path, label_path, emb_non_trainable=True):
        super().__init__(config=config)

        self.config = config
        seq_size = config['n_ctx']
        token_emb_dim = config['token_emb_dim']
        num_filters = config['num_filters']
        kernel_sizes = config['kernel_sizes']

        # glove embedding layer
        weights_matrix = super().load_embedding(embedding_path)
        self.embed = super().create_embedding_layer(weights_matrix, non_trainable=emb_non_trainable)
        
        # Densenet layer
        densenet_depth = config['densenet_depth']
        densenet_width = config['densenet_width']
        emb_dim = token_emb_dim
        densenet_first_num_filters = config['densenet_first_num_filters']
        densenet_num_filters = config['densenet_num_filters']
        densenet_last_num_filters = config['densenet_last_num_filters']
        self.densenet = DenseNet(densenet_depth, densenet_width, emb_dim, densenet_first_num_filters, densenet_num_filters, densenet_last_num_filters)
        self.layernorm_densenet = nn.LayerNorm(densenet_last_num_filters)

        # convolution layer
        self.textcnn = TextCNN(densenet_last_num_filters, num_filters, kernel_sizes)

        self.dropout = nn.Dropout(config['dropout'])

        # fully connected layer
        self.labels = super().load_label(label_path)
        label_size = len(self.labels)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, label_size)

    def forward(self, x):
        # x : [batch_size, seq_size]

        device = self.config['device']
        mask = torch.sign(torch.abs(x)).to(torch.uint8).to(device)
        # mask : [batch_size, seq_size]

        # 1. glove embedding
        embed_out = self.dropout(self.embed(x))
        # [batch_size, seq_size, token_emb_dim]

        # 2. DenseNet
        densenet_out = self.densenet(embed_out, mask)
        # densenet_out : [batch_size, seq_size, last_num_filters]
        densenet_out = self.layernorm_densenet(densenet_out)
        densenet_out = self.dropout(densenet_out)

        # 3. convolution
        textcnn_out = self.textcnn(densenet_out)
        # [batch_size, len(kernel_sizes) * num_filters]
        textcnn_out = self.dropout(textcnn_out)

        # 4. fully connected
        fc_out = self.fc(textcnn_out)
        # [batch_size, label_size]
        output = torch.softmax(fc_out, dim=-1)
        return output

class TextGloveDensenetDSA(BaseModel):
    def __init__(self, config, embedding_path, label_path, emb_non_trainable=True):
        super().__init__(config=config)

        self.config = config
        seq_size = config['n_ctx']
        token_emb_dim = config['token_emb_dim']

        # glove embedding layer
        weights_matrix = super().load_embedding(embedding_path)
        self.embed = super().create_embedding_layer(weights_matrix, non_trainable=emb_non_trainable)
        
        # Densenet layer
        densenet_depth = config['densenet_depth']
        densenet_width = config['densenet_width']
        emb_dim = token_emb_dim
        densenet_first_num_filters = config['densenet_first_num_filters']
        densenet_num_filters = config['densenet_num_filters']
        densenet_last_num_filters = config['densenet_last_num_filters']
        self.densenet = DenseNet(densenet_depth, densenet_width, emb_dim, densenet_first_num_filters, densenet_num_filters, densenet_last_num_filters)
        self.layernorm_densenet = nn.LayerNorm(densenet_last_num_filters)

        # DSA(Dynamic Self Attention) layer
        dsa_num_attentions = config['dsa_num_attentions']
        dsa_input_dim = densenet_last_num_filters
        dsa_dim = config['dsa_dim']
        dsa_r = config['dsa_r']
        self.dsa = DSA(config, dsa_num_attentions, dsa_input_dim, dsa_dim, dsa_r=dsa_r)
        self.layernorm_dsa = nn.LayerNorm(dsa_num_attentions * dsa_dim)

        self.dropout = nn.Dropout(config['dropout'])

        # fully connected layer
        self.labels = super().load_label(label_path)
        label_size = len(self.labels)
        self.fc = nn.Linear(dsa_num_attentions * dsa_dim, label_size)

    def forward(self, x):
        # x : [batch_size, seq_size]

        device = self.config['device']
        mask = torch.sign(torch.abs(x)).to(torch.uint8).to(device)
        # mask : [batch_size, seq_size]

        # 1. glove embedding
        embed_out = self.dropout(self.embed(x))
        # [batch_size, seq_size, token_emb_dim]

        # 2. DenseNet
        densenet_out = self.densenet(embed_out, mask)
        # densenet_out : [batch_size, seq_size, densenet_last_num_filters]
        densenet_out = self.layernorm_densenet(densenet_out)
        densenet_out = self.dropout(densenet_out)

        # 3. DSA(Dynamic Self Attention)
        dsa_out = self.dsa(densenet_out, mask) 
        # dsa_out : [batch_size, dsa_num_attentions * dsa_dim]
        '''
        dsa_out = self.layernorm_dsa(dsa_out)
        '''
        dsa_out = self.dropout(dsa_out)

        # 4. fully connected
        fc_out = self.fc(dsa_out)
        # [batch_size, label_size]
        output = torch.softmax(fc_out, dim=-1)
        return output

class TextBertCNN(BaseModel):
    def __init__(self, config, bert_config, bert_model, label_path, feature_based=False):
        super().__init__(config=config)

        self.config = config
        seq_size = config['n_ctx']
        num_filters = config['num_filters']
        kernel_sizes = config['kernel_sizes']
        fc_hidden_size = config['fc_hidden_size']

        # bert embedding layer
        self.bert_config = bert_config
        self.bert_model = bert_model
        hidden_size = bert_config.hidden_size
        self.feature_based = feature_based
        emb_dim = hidden_size

        # convolution layer
        self.textcnn = TextCNN(emb_dim, num_filters, kernel_sizes)

        self.dropout = nn.Dropout(config['dropout'])

        # fully connected layer
        self.labels = super().load_label(label_path)
        label_size = len(self.labels)
        self.layernorm1 = nn.LayerNorm(len(kernel_sizes) * num_filters)
        self.fc1 = nn.Linear(len(kernel_sizes) * num_filters, fc_hidden_size)
        self.layernorm2 = nn.LayerNorm(fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, label_size)

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
            # [batch_size, 0, hidden_size] corresponding to [CLS] == 'embedded[:, 0]'
        return embedded

    def forward(self, x):
        # x[0], x[1], x[2] : [batch_size, seq_size]

        # 1. bert embedding
        embedded = self.__compute_bert_embedding(x)
        # embedded : [batch_size, seq_size, hidden_size]
        embedded = self.dropout(embedded)

        # 2. convolution
        textcnn_out = self.textcnn(embedded)
        textcnn_out = self.layernorm1(textcnn_out)
        textcnn_out = self.dropout(textcnn_out)

        # 3. fully connected
        fc_hidden = self.fc1(textcnn_out)
        # fc_hidden : [batch_size, fc_hidden_size]
        fc_hidden = self.layernorm2(fc_hidden)
        fc_hidden = self.dropout(fc_hidden)
        fc_out = self.fc2(fc_hidden)
        # fc_out : [batch_size, label_size]

        output = torch.softmax(fc_out, dim=-1)
        # output : [batch_size, label_size]
        return output

class TextBertCLS(BaseModel):
    def __init__(self, config, bert_config, bert_model, label_path, feature_based=False):
        super().__init__(config=config)

        self.config = config
        seq_size = config['n_ctx']
        num_filters = config['num_filters']
        kernel_sizes = config['kernel_sizes']

        # bert embedding layer
        self.bert_config = bert_config
        self.bert_model = bert_model
        hidden_size = bert_config.hidden_size
        self.feature_based = feature_based

        self.dropout = nn.Dropout(config['dropout'])

        # fully connected layer
        self.labels = super().load_label(label_path)
        label_size = len(self.labels)
        self.fc = nn.Linear(hidden_size, label_size)

    def __compute_bert_embedding(self, x):
        if self.feature_based:
            # feature-based
            with torch.no_grad():
                bert_outputs = self.bert_model(input_ids=x[0],
                                               attention_mask=x[1],
                                               token_type_ids=x[2])
                pooled = bert_outputs[1]
        else:
            # fine-tuning
            # x[0], x[1], x[2] : [batch_size, seq_size]
            bert_outputs = self.bert_model(input_ids=x[0],
                                           attention_mask=x[1],
                                           token_type_ids=x[2])
            pooled = bert_outputs[1] # first token embedding([CLS]), see BertPooler
            # [batch_size, hidden_size]
        embedded = pooled
        return embedded

    def forward(self, x):
        # x[0], x[1], x[2] : [batch_size, seq_size]
        # 1. bert embedding
        embedded = self.__compute_bert_embedding(x)
        # embedded : [batch_size, hidden_size]
        embedded = self.dropout(embedded)

        # 2. fully connected
        output = torch.softmax(self.fc(embedded), dim=-1)
        # output : [batch_size, label_size]
        return output

