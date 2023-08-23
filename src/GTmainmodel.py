from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn


import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
import networkx as nx
from utils import *
from NAFSmodel2 import *
from torch.utils.data.dataloader import DataLoader
from GTmodel import *






class MultiheadAttention1(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiheadAttention1, self).__init__()
        set_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        assert in_dim % num_heads == 0
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.num_heads = num_heads
        self.depth = in_dim // num_heads
        self.out_dim = out_dim
        self.query_linear = nn.Linear(in_dim, in_dim)
        self.key_linear = nn.Linear(in_dim, in_dim)
        self.value_linear = nn.Linear(in_dim, in_dim)

        self.output_linear = nn.Linear(in_dim, out_dim )

    def split_heads(self, x, batch_size):
        # reshape input to [batch_size, num_heads, seq_len, depth]

        x_szie = x.size()[:-1] + (self.num_heads, self.depth)
        x = x.reshape(x_szie)
        # transpose to [batch_size, num_heads, depth, seq_len]
        return x.transpose(-1, -2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Split the inputs into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        # Apply mask (if necessary)
        if mask is not None:
            mask = mask.unsqueeze(1)  # add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=0)
        attention_output = torch.matmul(attention_weights, V)

        # Merge the heads
        output_size = attention_output.size()[:-2]+ (query.size(1),)
        attention_output = attention_output.transpose(-1, -2).reshape((output_size))

        # Linear projection to get the final output
        attention_output = self.output_linear(attention_output)

        return torch.sigmoid(attention_output)



class GraphTransformerLayer1(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, hidden_dim, fout_dim, num_heads, dropout, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()
        set_seed(args.seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fout_dim = fout_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiheadAttention1(in_dim, hidden_dim, num_heads)

        self.residual_layer1 = nn.Linear(in_dim, fout_dim)  #残差1
        self.residual_layer2 = nn.Linear(in_dim, fout_dim) #残差2

        self.O = nn.Linear(hidden_dim, fout_dim)


        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(fout_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(fout_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(fout_dim, fout_dim * 2)
        self.FFN_layer2 = nn.Linear(fout_dim * 2, fout_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(fout_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(fout_dim)


        self.hidden = nn.Linear(self.fout_dim + self.fout_dim, in_dim, dropout)
    def forward(self, h1,h2):
##异质网络的节点特征结合相似性网络的特征##
        h_in1 = self.residual_layer1(h1)  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(h1, h1, h1)

        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))

        if self.residual:
            attn_out = h_in1 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm1(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm1(attn_out)

        #h_in2 = attn_out  # for second residual connection

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)


###fea+SF
        h2 = torch.cat((attn_out, h2), dim=1)
        h2 = F.leaky_relu(self.hidden(h2))
        h_in2 = self.residual_layer2(h2)
        attn_out = self.attention(h2, h2, h2)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))
        if self.residual:
            attn_out = h_in2 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm2(attn_out)
        h_in3 = attn_out

        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)


        return attn_out






