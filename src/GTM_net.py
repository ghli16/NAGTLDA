import torch
import torch.nn as nn
import torch.nn.functional as F
from NAFSmodel2 import *
from GTmodel import *
import numpy as np
from utils import *

class GTM_net(nn.Module):
    def __init__(self, args, X):
        super().__init__()
        self.X = X

        set_seed(args.seed)
        #self.SF = torch.tensor(SF, requires_grad=False)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        x_dim = self.X.size(1)
        #pe_dim = self.SF.size(1)
        in_dim = args.in_dim
        out_dim = args.out_dim
        fout_dim = args.fout_dim
        head_num = args.head_num
        pos_enc_dim = args.pos_enc_dim
        dropout = args.dropout
        seed = args.seed
        self.L = args.L
        self.output = args.output_t
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.hidden = FC(x_dim, args.in_dim, dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(in_dim, out_dim, fout_dim, head_num,dropout,
                                                            self.layer_norm, self.batch_norm, self.residual) for _ in range(self.L - 1)])
        self.layers.append(
            GraphTransformerLayer(in_dim, out_dim, fout_dim, head_num, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.FN = nn.Linear(fout_dim, self.output)
        self.Bilinear = nn.Linear(self.output, self.output)
    def forward(self):
        X = F.leaky_relu(self.hidden(self.X))
        for conv in self.layers:
            h = conv(X)
        h = F.leaky_relu((self.FN(h)))

        return h



