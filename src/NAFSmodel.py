from __future__ import division
from __future__ import print_function
import csv
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



def run(train_matrix, args):
    set_seed(args.seed)
    if args.dataset == 'wiki':
        _, _, lncrna_sim, dis_sim= load_wiki(train_matrix)
        lncrna_sim = torch.FloatTensor(lncrna_sim )
        dis_sim = torch.FloatTensor(dis_sim)
        lncRNA_adj  = sp.csr_matrix(get_adjacency_matrix(lncrna_sim, 0.75))
        dis_adj  = sp.csr_matrix(get_adjacency_matrix(dis_sim, 0.75))
        ''' SF1 = S_main(miRNA_adj, miRNA_adj.shape[0], args)
        SF2 = S_main(dis_adj, dis_adj.shape[0], args)
        SF = torch.cat((SF1, SF2), dim=0)'''
        #y = np.array(SF.detach().numpy())

    n_nodes1 = lncRNA_adj.shape[0]
    n_nodes2 = dis_adj.shape[0]

    for hop in range(args.hops, args.hops+1):
        inputrna_features = 0.
        if args.dataset == 'pubmed':
            r_list = [0.3, 0.4, 0.5]
        else:
            r_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for r in r_list:
            adjrna_norm = normalize_adj(lncRNA_adj, r)

            featuresrna_list = []
            featuresrna_list.append(lncrna_sim)
            for _ in range(hop):
                featuresrna_list.append(torch.spmm(adjrna_norm, featuresrna_list[-1]))

            weight_list = []
            norm_fea = torch.norm(lncrna_sim, 2, 1).add(1e-10)
            for fea in featuresrna_list:
                norm_cur = torch.norm(fea, 2, 1).add(1e-10)

                temp = torch.div((lncrna_sim*fea).sum(1), norm_fea)
                temp = torch.div(temp, norm_cur)
                weight_list.append(temp.unsqueeze(-1))

            weight = F.softmax(torch.cat(weight_list, dim=1), dim=1)

            inputrna_feas = []
            for i in range(n_nodes1):
                fea = 0.
                for j in range(hop+1):
                    fea += (weight[i][j]*featuresrna_list[j][i]).unsqueeze(0)
                inputrna_feas.append(fea)
            inputrna_feas = torch.cat(inputrna_feas, dim=0)
            inputrna_features = inputrna_features + inputrna_feas
        inputrna_features /= len(r_list)
    for hop in range(args.hops, args.hops + 1):
        inputdis_features = 0.
        if args.dataset == 'pubmed':
                r_list = [0.3, 0.4, 0.5]
        else:
                r_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for r in r_list:
            adjdis_norm = normalize_adj(dis_adj, r)

            featuresdis_list = []
            featuresdis_list.append(dis_sim)
            for _ in range(hop):
                featuresdis_list.append(torch.spmm(adjdis_norm, featuresdis_list[-1]))

            weight_list = []
            norm_fea = torch.norm(dis_sim, 2, 1).add(1e-10)
            for fea in featuresdis_list:
                norm_cur = torch.norm(fea, 2, 1).add(1e-10)

                temp = torch.div((dis_sim * fea).sum(1), norm_fea)
                temp = torch.div(temp, norm_cur)
                weight_list.append(temp.unsqueeze(-1))

            weight = F.softmax(torch.cat(weight_list, dim=1), dim=1)

            inputdis_feas = []
            for i in range(n_nodes2):
                fea = 0.
                for j in range(hop + 1):
                    fea += (weight[i][j] * featuresdis_list[j][i]).unsqueeze(0)
                inputdis_feas.append(fea)
            inputdis_feas = torch.cat(inputdis_feas, dim=0)
            inputdis_features = inputdis_features + inputdis_feas
        inputdis_features /= len(r_list)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tm = torch.nn.Linear(inputrna_features.shape[1], inputdis_features.shape[1])
    inputrna_features = tm(inputrna_features)
    all_input_features = torch.cat((inputrna_features, inputdis_features), dim=0)
    all_input_features_m=all_input_features.detach().numpy()

    return all_input_features, lncRNA_adj, dis_adj



