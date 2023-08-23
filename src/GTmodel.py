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

parser = argparse.ArgumentParser()
parser.add_argument('--workers', default=8, type=int,help='Number of parallel processes.')
parser.add_argument('--weighted', action='store_true', default=False,help='Treat graph as weighted')
parser.add_argument('--epochs', default=100, type=int,help='The training epochs of SDNE')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss on embedding matrix')
parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
parser.add_argument('--alpha', default=1e-6, type=float, help='alhpa is a hyperparameter in SDNE')
parser.add_argument('--beta', default=5., type=float, help='beta is a hyperparameter in SDNE')
parser.add_argument('--nu1', default=1e-5, type=float, help='nu1 is a hyperparameter in SDNE')
parser.add_argument('--nu2', default=1e-4, type=float,  help='nu2 is a hyperparameter in SDNE')
parser.add_argument('--bs', default=500, type=int, help='batch size of SDNE')
parser.add_argument('--nhid0', default=1000, type=int, help='The first dim')
parser.add_argument('--nhid1', default=128, type=int, help='The second dim')
parser.add_argument('--step_size', default=150, type=int, help='The step size for lr')
parser.add_argument('--gamma', default=0.9, type=int,  help='The gamma for lr')

parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--dataset', type=str, default='wiki', help='type of dataset.')
parser.add_argument('--hops', type=int, default=7, help='number of hops.')
parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
parser.add_argument('--epoch', type=int, default=150, help='train_number.')
parser.add_argument('--in_dim', type=int, default=1024, help='in_feature.')
parser.add_argument('--out_dim', type=int, default=256, help='out_feature.')
parser.add_argument('--fout_dim', type=int, default=128, help='f-out_feature.')
parser.add_argument('--output_t', type=int, default=64, help='finally_out_feature.')
parser.add_argument('--head_num', type=int, default=8, help='head_number.')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout.')
parser.add_argument('--pos_enc_dim', type=int, default=64, help='pos_enc_dim.')
parser.add_argument('--residual', type=bool, default=True, help='RESIDUAL.')
parser.add_argument('--layer_norm', type=bool, default=True, help='LAYER_NORM.')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch_norm.')
parser.add_argument('--L', type=int, default=10, help='TransformerLayer.')


parser.add_argument('--in_dim1', type=int, default=2048, help='in_feature.')
parser.add_argument('--out_dim1', type=int, default=256, help='out_feature.')
parser.add_argument('--fout_dim1', type=int, default=128, help='f-out_feature.')
parser.add_argument('--output_t1', type=int, default=64, help='finally_out_feature.')
parser.add_argument('--head_num1', type=int, default=64, help='head_number.')
parser.add_argument('--pos_enc_dim1', type=int, default=64, help='pos_enc_dim.')
parser.add_argument('--residual1', type=bool, default=True, help='RESIDUAL.')
parser.add_argument('--layer_norm1', type=bool, default=True, help='LAYER_NORM.')
parser.add_argument('--batch_norm1', type=bool, default=False, help='batch_norm.')
parser.add_argument('--L1', type=int, default=20, help='TransformerLayer.')


args = parser.parse_args()



class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiheadAttention, self).__init__()
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

        self.output_linear = nn.Linear(in_dim, out_dim)

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



class GraphTransformerLayer(nn.Module):
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

        self.attention = MultiheadAttention(in_dim, hidden_dim, num_heads)

        self.residual_layer1 = nn.Linear(in_dim, fout_dim)  #残差


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

    def forward(self, h):
        h_in1 = self.residual_layer1(h)  # for first residual connection
        #mask1 = torch.ones((653,32,32))
        # multi-head attention out
        attn_out = self.attention(h, h, h)
        #h = attn_out.view(-1, self.out_channels)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))

        if self.residual:
            attn_out = h_in1 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm1(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm1(attn_out)

        h_in2 = attn_out  # for second residual connection

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)

        if self.residual:
            attn_out = h_in2 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm2(attn_out)

        #print(torch.sum(torch.gt(attn_out, 0)))
        return attn_out






