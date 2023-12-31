from __future__ import division
from __future__ import print_function

import argparse
import time
from SDNE_M import *
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from utils import *
from GTmodel import *

class FC(nn.Module):
    def __init__(self, nfeat, nout, dropout):
        super(FC, self).__init__()

        self.fcn = nn.Linear(nfeat, nout)

        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.fcn(x)

        return F.leaky_relu(x)

class FC1(nn.Module):
    def __init__(self, pos_enc_dim, nout, dropout):
        super(FC1, self).__init__()


        self.fcn = nn.Linear(pos_enc_dim, nout)

        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.fcn(x)

        return F.leaky_relu(x)


def run2(train_matrix, args):
    set_seed(args.seed)
    if args.dataset == 'wiki':
        features, adj, _, _= load_wiki(train_matrix)

    n_nodes, feat_dim = features.shape

    adj_orig = adj
    adj_orig = adj_orig - \
        sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [
                      0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()


    for hop in range(args.hops, args.hops+1):
        input_features = 0.
        if args.dataset == 'pubmed':
            r_list = [0.3, 0.4, 0.5]
        else:
            r_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for r in r_list:
            adj_norm = normalize_adj(adj, r)

            features_list = []
            features_list.append(features)
            for _ in range(hop):
                features_list.append(torch.spmm(adj_norm, features_list[-1]))

            weight_list = []
            norm_fea = torch.norm(features, 2, 1).add(1e-10)
            for fea in features_list:
                norm_cur = torch.norm(fea, 2, 1).add(1e-10)

                temp = torch.div((features*fea).sum(1), norm_fea)
                temp = torch.div(temp, norm_cur)
                weight_list.append(temp.unsqueeze(-1))

            weight = F.softmax(torch.cat(weight_list, dim=1), dim=1)

            input_feas = []
            for i in range(n_nodes):
                fea = 0.
                for j in range(hop+1):
                    fea += (weight[i][j]*features_list[j][i]).unsqueeze(0)
                input_feas.append(fea)
            input_feas = torch.cat(input_feas, dim=0)
            input_features = input_features + input_feas
        input_features /= len(r_list)
        input_features_data = input_features.detach().numpy()
        '''with open('../features matrix/middle1.csv', 'w', newline='') as csvfile:
            tensor_writer = csv.writer(csvfile)
            for row in input_features_data:
                tensor_writer.writerow(row)'''
        '''sim = torch.sigmoid(torch.mm(input_features, input_features.T))

        roc_score, ap_score = get_roc_score(sim.numpy(), adj_orig, test_edges, test_edges_false)
        print(f'AUC: {roc_score:.4f}, AP: {ap_score:.4f}, Hop: {hop:02d}')'''

    return input_features



