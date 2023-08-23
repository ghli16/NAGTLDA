import torch
import numpy as np
import argparse
from utils import *
from MNN import *
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

def S_loss(X, Node, args):
    set_seed(args.seed)
    X = sp.csr_matrix(X).toarray()  # 将稀疏矩阵转换为稠密矩阵
    X = torch.FloatTensor(X)
    model_s = MNN(Node, args.nhid0, args.nhid1, args.dropout, args.alpha)
    Data = Dataload(X, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True )
    loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
    for index in Data:
        adj_batch = X[index]
        adj_mat = adj_batch[:, index]
        b_mat = torch.ones_like(adj_batch)
        b_mat[adj_batch != 0] = args.beta
        L_1st, L_2nd, L_all = model_s(adj_batch, adj_mat, b_mat)
        L_reg = 0
        for param in model_s.parameters():
            L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
        Loss = L_all + L_reg



    return Loss