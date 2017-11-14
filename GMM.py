# -*- coding: utf-8 -*-
# Author: 小狼狗
'''
=================================================================
                        EM算法计算Gmm
=================================================================
'''
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gmm(X, K):
    threshold  = 1e-15
    N,D = np.shape(X)
    randV = np.random.randint(1,N,K)         # 选取初始化点，从1~N中选取K个数
    centroids = X[randV]                     # 随机选取K个点作为μ的初始向量
    pMiu,pPi,pSigma = inti_params(centroids, K, X, N, D)    #初始化参数
    L_prev = 0
    while True:
        #E Step
        Px = calc_prop(X, N, K, pMiu, pSigma, threshold, D)
        pGamma = Px * np.tile(pPi, (N,1))   #对应元素相乘  80*4
        pGamma = pGamma / np.tile((np.sum(pGamma,axis=1)),(K,1)).T
        #M Step
        Nk = np.sum(pGamma,axis=0)
        pMiu = np.dot(np.dot(np.diag(1 / Nk),pGamma.T),X)  #4*2
        pPi = Nk / N
        for kk in range(K):
            Xshift = X - np.tile(pMiu[kk],(N,1))
            pSigma[:,:,kk] = (np.dot(np.dot(Xshift.T,np.diag(pGamma[:,kk])),Xshift)) / Nk[kk]

        #验证是否收敛
        L = np.sum(np.log(np.dot(Px, pPi.T)))
        if L-L_prev < threshold:
            break
        L_prev = L

    return Px

#初始化参数
def inti_params(centroids, K, X, N, D):
    pMiu = centroids
    pPi = np.zeros((1,K))   #[[ 0.  0.  0.  0.]]
    pSigma = np.zeros((D,D,K))
    #分别计算每个样本与各个中心的距离 80*4
    distmat = np.tile(np.sum(X * X,axis=1),(K,1)).T \
                                       + np.tile(np.sum(pMiu * pMiu,axis = 1).T,(N,1)) \
                                                              - 2 * np.dot(X, pMiu.T)  #80*4

    labels = np.argmin(distmat,axis=1)     #找出每个数据离哪个中心最近，分别标注类别 0,1,2 or 3

    for k in range(K):
        X_k = X[labels==k]
        pPi[0][k] = float(np.shape(X_k)[0]) / N     # 样本数除以 N 得到初始化概率π_k
        pSigma[:,:,k] = np.cov(X_k.T)
    return pMiu,pPi,pSigma

#计算概率
def calc_prop(X, N, K, pMiu, pSigma, threshold, D):
    Px = np.zeros((N, K))
    for k in range(K):
        Xshift = X - np.tile(pMiu[k],(N,1))
        inv_pSigma = np.linalg.inv(pSigma[:,:,k]) \
                                    + np.diag(np.tile(threshold,(1,np.ndim(pSigma[:,:,k]))))
        tmp = np.sum(np.dot(Xshift,inv_pSigma) * Xshift,axis=1)
        coef = (2*np.pi)**(-D/2) * np.sqrt(np.linalg.det(inv_pSigma))  #矩阵行列式的倒数 = 逆的行列式
        Px[:,k] = coef * np.exp(-0.5 * tmp)
    return Px

#载入数据
def loadfile(filename):
    lst = []
    with open(filename, 'r') as f:
        data = csv.reader(f)
        for row in data:
            lst.append(row)
    f.close()
    for i in range(len(lst)):
        lst[i] = [float(x) for x in lst[i]]
    return lst

if __name__ == '__main__':
    X = loadfile('test_set.csv')
    X = np.array(X)
    ppx = gmm(X,4)
    index = np.argmax(ppx,axis=1)
    plt.figure()
    plt.scatter(X[index==0][:,0],X[index==0][:,1],s=60,c=u'r',marker=u'o')
    plt.scatter(X[index==1][:,0],X[index==1][:,1],s=60,c=u'b',marker=u'o')
    plt.scatter(X[index==2][:,0],X[index==2][:,1],s=60,c=u'y',marker=u'o')
    plt.scatter(X[index==3][:,0],X[index==3][:,1],s=60,c=u'g',marker=u'o')
    plt.show()


