from GTM_net import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from NAFSmodel2 import *
import numpy as np
from GTmodel import *
from GTmainmodel import *
import csv
class main_GT(nn.Module):
    def __init__(self, args, features, mirna_adj, dis_adj , x1, A):
        super().__init__()
        set_seed(args.seed)
        self.A =A
        self.x1 = x1.clone().detach().requires_grad_(False)
        mirna_adj = sp.csr_matrix(mirna_adj).toarray()  # 将稀疏矩阵转换为稠密矩阵
        self.mirna_adj = torch.FloatTensor(mirna_adj)

        dis_adj = sp.csr_matrix(dis_adj).toarray()  # 将稀疏矩阵转换为稠密矩阵
        self.dis_adj = torch.FloatTensor(dis_adj)

        #self.x1 = torch.tensor(x1, dtype=float, requires_grad=False)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.model = GTM_net(args, x1)
        Node1 = self.mirna_adj.shape[0]
        Node2 = self.dis_adj.shape[0]
        self.S1 = MNN(Node1, args.nhid0, args.nhid1, args.dropout, args.alpha)
        self.S2 = MNN(Node2, args.nhid0, args.nhid1, args.dropout, args.alpha)

        self.features = features.clone().detach().requires_grad_(True)
        #self.features = torch.tensor(features)

        #self.SF = torch.tensor(SF, dtype=float, requires_grad=False)
        x_dim = args.output_t
        fea_dim = self.features.size(1)
        sf_dim = args.nhid1

        in_dim = args.in_dim1
        out_dim = args.out_dim1
        fout_dim = args.fout_dim1
        head_num = args.head_num1
        pos_enc_dim = args.pos_enc_dim
        dropout = args.dropout
        seed = args.seed



        self.L = args.L1
        self.output = args.output_t1
        self.layer_norm = args.layer_norm1
        self.batch_norm = args.batch_norm1
        self.residual = args.residual1

        '''self.f_fn = nn.Linear(fea_dim, args.output_t, dropout)
        self.sf_fn = nn.Linear(sf_dim, args.output_t, dropout)
        self.hidden = nn.Linear(x_dim, in_dim, dropout)'''

        self.f_fn = nn.Linear(fea_dim, sf_dim, dropout)
        self.hidden = nn.Linear(x_dim + sf_dim, in_dim, dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer1(in_dim, out_dim, fout_dim, head_num,dropout,
                                                            self.layer_norm, self.batch_norm, self.residual) for _ in range(self.L - 1)])
        self.layers.append(
            GraphTransformerLayer1(in_dim, out_dim, fout_dim, head_num, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.FN = nn.Linear(fout_dim, self.output)
        self.Bilinear = nn.Linear(self.output, self.output)

        self.W_att = nn.Parameter(torch.randn(x_dim, x_dim))
        self.V_att = nn.Parameter(torch.randn(x_dim, 1))

        self.ff1 = nn.Linear(fea_dim, 64)
        self.ff2 = nn.Linear(64, fea_dim)
        self.xx1 = nn.Linear(x_dim, 16)
        self.xx2 = nn.Linear(16, x_dim)
        self.sf_fn = nn.Linear(sf_dim, fout_dim)
    def forward(self):

        X = self.model()
        x_m = X.detach().numpy()
        with open('../features matrix/T1.csv', 'w', newline='') as csvfile:
            tensor_writer = csv.writer(csvfile)
            for row in x_m:
                tensor_writer.writerow(row)
        SF1 = self.S1.savector(self.mirna_adj)
        SF2 = self.S2.savector(self.dis_adj)
        SF = torch.cat((SF1, SF2), dim=0)
        SF = self.sf_fn(SF)
        features = self.f_fn(self.features)
        #aux = SF + features
        h1 = torch.cat((X, features), dim=1)  # 初步的学习到的节点信息和结构信息进行融合，融合的结果和transformer得到的异质图的节点特征信息进行融合成为AF
        h2 = SF

        h1 = F.leaky_relu(self.hidden(h1))
        for conv in self.layers:
            h = conv(h1, h2)
        h = F.leaky_relu((self.FN(h)))
        h_m=h.detach().numpy()
        hl = h[:self.A.shape[0]]
        hd = h[self.A.shape[0]:]
        H1 = self.Bilinear(hl)
        H = torch.sigmoid(torch.matmul(H1, hd.T))

        return H



