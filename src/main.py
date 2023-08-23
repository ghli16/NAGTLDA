import numpy as np
import torch
from GTmodel import *
from GTM_net import *
from clac_metric import get_metrics
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import gc
from utils import *
from NAFSmodel2 import *
from NAFSmodel import *
from main_model import *
import matplotlib.pyplot as plt
import scipy
def train(model, train_matrix,optimizer,mirna_adj, dis_adj, args):
    model.train()
    criterion = torch.nn.BCELoss(reduction='sum')
    def train_epoch():
        optimizer.zero_grad()  # 将所有模型参数的梯度置为0
        score = model()
        loss = (criterion(score, torch.FloatTensor(train_matrix)))+S_loss(mirna_adj, mirna_adj.shape[0], args)+S_loss(dis_adj, dis_adj.shape[0], args)
        loss = loss.requires_grad_()
        loss.backward()  # 反向传播得到每个参数的梯度值
        optimizer.step()  # 通过梯度下降执行梯度更新
        return loss

    for epoch in range(1, args.epoch + 1):
        train_reg_loss = train_epoch()

        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass



def PredictScore(features,mirna_adj, dis_adj, x1, train_matrix, args):
    set_seed(args.seed)
    model = main_GT(args, features, mirna_adj, dis_adj, x1, train_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
    '''print(model)
    for parameters in model.parameters():
        print(parameters, ':', parameters.size())'''
    train(model, train_matrix, optimizer, mirna_adj, dis_adj, args)
    return model()

##positive : negative = random
'''def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    set_seed(args.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]
    return index'''
##positive : negative = 1 : 1
def balance_samples(pos_index, neg_index):

    pos_num = len(pos_index)
    neg_num = len(neg_index)
    if pos_num > neg_num:
        # 对正样本进行下采样
        balanced_pos_index = random.sample(pos_index, neg_num)
        balanced_neg_index = neg_index
    else:
        # 对负样本进行下采样
        balanced_pos_index = pos_index
        balanced_neg_index = random.sample(neg_index, pos_num)
    return balanced_pos_index, balanced_neg_index
def random_index(index_matrix, sizes):
    set_seed(sizes.seed)
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp
def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    set_seed(args.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = []
    for i in range(sizes.k_fold):
        # 对每一折的正负样本进行平衡处理
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index
'''def balance_samples(pos_index, neg_index):
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    if pos_num > neg_num:
        # 对正样本进行下采样
        balanced_pos_index = random.sample(pos_index, neg_num)
        balanced_neg_index = neg_index
    else:
        # 对负样本进行下采样
        balanced_pos_index = pos_index
        balanced_neg_index = random.sample(neg_index, pos_num)
    return balanced_pos_index, balanced_neg_index
def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp
def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = []
    for i in range(sizes.k_fold):
        # 对每一折的正负样本进行平衡处理
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index'''
##positive : negative = 1 : 5
'''def balance_samples(pos_index, neg_index):
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    if pos_num > 5*neg_num:
        # 对正样本进行下采样
        balanced_pos_index = random.sample(pos_index, 5*neg_num)
        balanced_neg_index = neg_index
    else:
        # 对负样本进行下采样
        balanced_pos_index = pos_index
        balanced_neg_index = random.sample(neg_index, int(pos_num/5)+1)
    return balanced_pos_index, balanced_neg_index

def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp
def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = []
    for i in range(sizes.k_fold):
        # 对每一折的正负样本进行平衡处理
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index'''
##positive : negative = 1 : 10
'''def balance_samples(pos_index, neg_index):
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    if pos_num > 10 * neg_num:
        # 对正样本进行下采样
        balanced_pos_index = random.sample(pos_index, 10 * neg_num)
        balanced_neg_index = neg_index
    else:
        # 对负样本进行下采样
        balanced_pos_index = pos_index
        balanced_neg_index = random.sample(neg_index, int(pos_num/10))
    return balanced_pos_index, balanced_neg_index

def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    set_seed(args.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    balanced_pos_index = []
    balanced_neg_index = []
    for i in range(sizes.k_fold):
        # 对每一折的正负样本进行平衡处理
        bp, bn = balance_samples(pos_index[i], neg_index[i])
        balanced_pos_index.append(bp)
        balanced_neg_index.append(bn)
    index = [balanced_pos_index[i] + balanced_neg_index[i] for i in range(sizes.k_fold)]
    return index'''

def cross_validation_experiment(A, args):
    index = crossval_index(A, args)
    metric = np.zeros((1, 7))
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    pre_matrix = np.zeros(A.shape)
    print("seed=%d, evaluating lncRNA-disease...." % (args.seed))
    for k in range(args.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(A, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0  # 将5折中的一折变为0
        drug_len = A.shape[0]
        dis_len = A.shape[1]
        features, mirna_adj, dis_adj = run(train_matrix, args)                               
        x1= run2(train_matrix, args)
        drug_mic_res = PredictScore(
                features, mirna_adj, dis_adj, x1, train_matrix, args)  # 预测得到的关联矩阵
        predict_y_proba = drug_mic_res.reshape(drug_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]  #从预测分数矩阵中取出验证集的预测结果 只返回相应的预测分数
        A = np.array(A)
        metric_tmp = get_metrics(A[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])   # 预测结果所得的评价指标
        fpr, tpr, t = roc_curve(A[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])
        precision, recall, _ = precision_recall_curve(A[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        auprs.append(metric_tmp[1])
        print(metric_tmp)
        metric += metric_tmp      #  五折交叉验证的结果求和
        del train_matrix  # del只删除变量，不删除数据
        gc.collect()  # 垃圾回收
    print('Mean:', metric / args.k_fold)
    metric = np.array(metric / args.k_fold)   #  五折交叉验证的结果求均值
    return metric, pre_matrix, drug_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs


def main(args):
    set_seed(args.seed)
    results = []
    data_path = '../dataset/'
    data_set = 'Raw_Dataset/'


    lncRNA_disease = np.loadtxt(data_path + data_set + 'lncRNA-disease.txt', delimiter=' ')

    A = np.matrix(lncRNA_disease, copy=True)
    result, pre_matrix, drug_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs = cross_validation_experiment(A, args)

    sizes = Sizes(drug_len, dis_len)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
    plot_auc_curves(fprs, tprs, aucs)
    plot_prc_curves(precisions, recalls, auprs)


if __name__== '__main__':
    main(args)

