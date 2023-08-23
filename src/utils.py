import pickle as pkl
import csv
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.sparse as sp
from torch.utils import data
from torch.utils.data import DataLoader
import dgl
import random
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import sklearn.preprocessing as preprocess
from munkres import Munkres
from sklearn import metrics


def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_wiki(train_matrix):
    data_path = '../dataset/'
    data_set = 'Raw_Dataset/'

    #miRNA_fun_sim = np.loadtxt(data_path + data_set + 'lncRNA-lncRNA.txt', delimiter=' ')
    disease_sem_sim = np.loadtxt(data_path + data_set + 'disease-disease.txt', delimiter=' ')
    lncRNA_disease = np.loadtxt(data_path + data_set + 'lncRNA-disease.txt', delimiter=' ')
    #lncRNA_fun_sim = np.loadtxt(data_path + data_set + 'lnc2017.csv', delimiter=',')
    # disease_sem_sim = np.loadtxt(data_path + data_set + 'disSem.csv', delimiter=',')
    # lncRNA_disease = np.loadtxt(data_path + data_set + 'A.csv', delimiter=',')

    lncRNA_disease_matrix = np.matrix(lncRNA_disease, copy=True)  #邻接矩阵

    lncRNA_fun_sim = fun_Sim(train_matrix, disease_sem_sim) #重算功能相似性
    lncRNA_sim, disease_sim = get_syn_sim(train_matrix, lncRNA_fun_sim, disease_sem_sim, mode=1) # 重算高斯核相似性

    """对称初始特征矩阵"""
    miRNA_matrix1 = np.matrix(
        np.zeros((train_matrix.shape[0], train_matrix.shape[0]), dtype=np.int8))
    dis_matrix1 = np.matrix(
        np.zeros((train_matrix.shape[1], train_matrix.shape[1]), dtype=np.int8))

    mat11 = np.hstack((miRNA_matrix1, train_matrix))
    mat21 = np.hstack((train_matrix.T, dis_matrix1))
    features = np.vstack((mat11, mat21))
    features = torch.FloatTensor(features)
    features_m=features.detach().numpy()
    with open('../features matrix/initial.csv', 'w', newline='') as csvfile:
        tensor_writer = csv.writer(csvfile)
        for row in features_m:
            tensor_writer.writerow(row)

    """构造异构网络"""
    mat12 = np.hstack((lncRNA_sim, train_matrix))
    mat22 = np.hstack((train_matrix.T, disease_sim))
    adj = np.vstack((mat12, mat22))
    adj = sp.csr_matrix(adj)
    return features, adj, lncRNA_sim, disease_sim


class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.c = 12

def fun_Sim(circ_dis_matrix, dis_matrix):
    rows = circ_dis_matrix.shape[0]
    result = np.zeros((rows, rows))
    for i in range(0, rows):
        idx = np.where(circ_dis_matrix[i, :] == 1)
        if (np.size(idx,1)==0):
            continue
        for j in range(0, i+1):
            idy = np.where(circ_dis_matrix[j, :] == 1)
            if (np.size(idy,1)==0):
                continue
            sum1 = 0
            sum2 = 0
            for k1 in range(0, np.size(idx,1)):
                max1 = 0
                for m in range(0, np.size(idy,1)):
                    if (dis_matrix[idx[0][k1], idy[0][m]]>max1):
                        max1 = dis_matrix[idx[0][k1], idy[0][m]]
                sum1 = sum1 + max1
            for k2 in range(0, np.size(idy,1)):
                max2 = 0
                for n in range(0, np.size(idx, 1)):
                    if (dis_matrix[idx[0][n], idy[0][k2]] > max2):
                        max2 = dis_matrix[idx[0][n], idy[0][k2]]
                sum2 = sum2 + max2
            result[i, j] = (sum1 + sum2) / (np.size(idx,1) + np.size(idy,1))
            result[j, i] = result[i, j]
        for k in range(0, rows):
            result[k, k] = 1
    return result




def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape




def normalize_adj(mx, r):  # 求邻接矩阵被r参数化后的 邻接矩阵的规则化结果
    """Row-normalize sparse matrix"""
    mx = sp.coo_matrix(mx) + sp.eye(mx.shape[0])  # 邻接矩阵加上自连接
    rowsum = np.array(mx.sum(1))  # 求节点的度
    r_inv_sqrt_left = np.power(rowsum, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(rowsum, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_normalized = mx.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node
    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat
    def __len__(self):
        return self.Node






import numpy as np
#########################################高斯核相似性的计算###################################################
def get_syn_sim(A, seq_sim, str_sim, mode):    ############## 调用这个

    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_c = np.zeros((A.shape[0], A.shape[0]))
    syn_d = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if seq_sim[i, j] == 0:
                syn_c[i, j] = GIP_c_sim[i, j]
            else:
                syn_c[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2


    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if str_sim[i, j] == 0:
                syn_d[i, j] = GIP_d_sim[i, j]
            else:
                syn_d[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2
    return syn_c, syn_d

def GIP_kernel(Asso_RNA_Dis):       ####### 高斯核相似性核心代码
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def getGosiR(Asso_RNA_Dis):
    # calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r


#SR, SD = get_syn_sim(AM, SR, SD, mode=1)


#计算邻接矩阵
def get_adjacency_matrix(similarity_matrix, threshold):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


'''def get_adjacency_matrix(feat, k):
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C'''

def plot_auc_curves(fprs, tprs, aucs):
    mean_fpr = np.linspace(0, 1, 1000)
    tpr = []
    #plt.style.use('ggplot')
    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.8, label='ROC fold %d (AUC = %.4f)' % (i + 1, aucs[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    auc_std = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))
    filename = "../PT/m_f.csv"

       # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
           # 创建CSV写入器
           writer = csv.writer(csvfile)
           # 写入列表中的每个元素
           for item in mean_fpr:
               writer.writerow([item])

    filename = "../PT/m_t.csv"
       # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
           # 创建CSV写入器
           writer = csv.writer(csvfile)
           # 写入列表中的每个元素
           for item in mean_tpr:
               writer.writerow([item])
    plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle='--', color='navy', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower right', prop={"size": 8})
    #plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    left, bottom, width, height = [0.4, 0.4, 0.3, 0.3]
    ax1 = plt.axes([left, bottom, width, height])
    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        ax1.plot(fprs[i], tprs[i])
    ax1.plot(mean_fpr, mean_tpr)
    xmin, xmax = 0.1, 0.25  # x轴范围
    ymin, ymax = 0.85, 0.95  # y轴范围
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.show()


def plot_prc_curves(precisions, recalls, auprs):
    mean_recall = np.linspace(0, 1, 1000)
    precision = []
    #plt.style.use('ggplot')
    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.8, label='ROC fold %d (AUPR = %.4f)' % (i + 1, auprs[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(auprs)
    prc_std = np.std(auprs)
    plt.plot(mean_recall, mean_precision, color='b', alpha=0.8,
             label='Mean AUPR (AUPR = %.4f $\pm$ %.4f)' % (mean_prc, prc_std))  # AP: Average Precision
    filename = "../PT/m_r.csv"
    # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        writer = csv.writer(csvfile)
        # 写入列表中的每个元素
        for item in mean_recall:
            writer.writerow([item])

    filename = "../PT/m_p.csv"
    # 打开文件，以写入模式写入数据
    with open(filename, 'w', newline='') as csvfile:
        # 创建CSV写入器
        writer = csv.writer(csvfile)
        # 写入列表中的每个元素
        for item in mean_precision:
            writer.writerow([item])
    plt.plot([-0.05, 1.05], [1.05, -0.05], linestyle='--', color='navy', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower left', prop={"size": 8})
    #plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')

    left, bottom, width, height = [0.3, 0.4, 0.3, 0.3]
    ax1 = plt.axes([left, bottom, width, height])
    for i in range(len(recalls)):
        precision.append(np.interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        ax1.plot(recalls[i], precisions[i])
    ax1.plot(mean_recall, mean_precision)
    xmin, xmax = 0.83, 0.9  # x轴范围
    ymin, ymax = 0.85, 0.9  # y轴范围
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.show()









