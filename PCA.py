import csv
import time
import numpy as np
from sklearn.svm import SVC
from sklearn import grid_search
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
'''''''''''''''''
参数说明：
n_components:
意义：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
类型：int 或者 string，缺省时默认为None，所有成分被保留。
          赋值为int，比如n_components=1，将把原始数据降到一个维度。
          赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
copy:
类型：bool，True或者False，缺省时默认为True。
意义：表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，
因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算。
whiten:
类型：bool，缺省时默认为False
'''''''''''''''''''''''
# lst =[]
# with open('data1.csv','r') as f:
#     data = csv.reader(f)
#     for row in data:
#         lst.append(row)
# lst =np.reshape(lst,[150,11])
# X = lst[:,1:11]
# y = lst[:,0]
# pca = PCA(n_components=3)
#
# pca.fit(X)
# new_X = pca.fit_transform(X)
# #print(X)
# print(new_X)
# print(pca.explained_variance_ratio_)
# print(pca.n_components)
# X_train, X_test, y_train, y_test = train_test_split(
#                            new_X,y,test_size=0.2)
#
# parameter = {'kernel':("linear","rbf"), "C":[1, 10, 100]}
# svm = SVC()
# t1 = time.time()
# clf = grid_search.GridSearchCV(svm, parameter)
# clf.fit(X_train,y_train)
# t2 = time.time()
# pred = clf.predict(X_test)
# print("Training time:",round(t2-t1,3))
# print (clf.best_score_,clf.best_params_)
# print('accurancy:',clf.score(X_test, y_test))
# plt.scatter(new_X[:, 0], new_X[:, 1], c=y)
# plt.xlabel('N', fontsize=14)
# plt.ylabel('PRT', fontsize=14)
# plt.show()

#=============================================================================================
#自己实现
C = np.mat( "81,41,250,1.980,6.100,9.000,1,12.000,11.110,2.440;"
            "80,42,238,1.910,5.670,0.000,0,0.000,0.000,0.000;"
            "81,26,196,3.120,7.540,9.800,2,15.000,12.040,6.730;"
            "125,63,368,1.980,5.840,20.000,1,18.000,16.000,1.590;"
            "146,45,350,3.240,7.780,42.800,3,43.000,29.280,6.110")

C_mean = C.mean(axis=0)
c = C - C_mean
#print(c)

U,Sigma,V = np.linalg.svd(c.T, full_matrices=0)


S = np.diag(Sigma)
print(Sigma)
#print(type(U))
print(S[:3,]*V)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(C)
x = pca.fit_transform(C)
#y = pca.explained_variance_
print(x)





