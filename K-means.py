# -*- coding: utf-8 -*-
# Author: 小狼狗

from copy import deepcopy
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('./data/xclara.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=4)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)   #初始化类的中心点
print("Initial Centroids")
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)   #3*2
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)      #计算样本点与每个中心的距离
        cluster = np.argmin(distances) #将样本点划分到距离最近的类别中（0,1,2）
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]  #统计每个类别中的点集
        C[i] = np.mean(points, axis=0)      #更新该类的聚类中心
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=4, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

plt.show()

'''
===========================================================
                    scikit-learn
===========================================================
'''

# from sklearn.cluster import KMeans
#
# # Number of clusters
# kmeans = KMeans(n_clusters=3)
# # Fitting the input data
# kmeans = kmeans.fit(X)
# # Getting the cluster labels
# labels = kmeans.predict(X)
# # Centroid values
# centroids = kmeans.cluster_centers_
#
# # Comparing with scikit-learn centroids
# print("Centroid values")
# print("Scratch")
# print(C) # From Scratch
# print("sklearn")
# print(centroids) # From sci-kit learn

'''
============================================================
                             向量写法
============================================================
'''
def k_means(X,K):
    N, D = np.shape(X)
    randV = np.random.randint(1, N, K)  # 选取初始化点，从1~N中选取K个数
    centroids = X[randV]  #4*2
    l = 1
    while l > 1e-5:
        pMiu = np.copy(centroids)
        #分别计算每个样本与各个中心的距离 80*4
        distmat = np.tile(np.sum(X * X,axis=1),(K,1)).T \
                                       + np.tile(np.sum(pMiu * pMiu,axis = 1).T,(N,1)) \
                                       - 2 * np.dot(X, pMiu.T)  #80*4

        index = np.argmin(distmat,axis=1)     #找出每个数据离哪个中心最近，分别标注类别 0,1,2 or 3
        new_pMiu = []
        for k in range(K):
            X_k = X[index==k]
            centroids[k] = np.mean(X_k, axis=0)
        l = np.sum(np.linalg.norm(centroids - pMiu, axis=1))


    plt.figure()
    plt.scatter(X[index == 0][:, 0], X[index == 0][:, 1], s=60, c=u'r', marker=u'o')
    plt.scatter(X[index == 1][:, 0], X[index == 1][:, 1], s=60, c=u'b', marker=u'o')
    plt.scatter(X[index == 2][:, 0], X[index == 2][:, 1], s=60, c=u'y', marker=u'o')
    plt.scatter(X[index == 3][:, 0], X[index == 3][:, 1], s=60, c=u'g', marker=u'o')
    plt.scatter(centroids[0][0], centroids[0][1], s=60, c=u'black', marker=u'*')
    plt.scatter(centroids[1][0], centroids[1][1], s=60, c=u'black', marker=u'*')
    plt.scatter(centroids[2][0], centroids[2][1], s=60, c=u'black', marker=u'*')
    plt.scatter(centroids[3][0], centroids[3][1], s=60, c=u'black', marker=u'*')
    plt.show()

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
    X = loadfile('./data/test_set.csv')
    X = np.array(X)
    k_means(X, 4)