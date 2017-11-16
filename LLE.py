# -*- coding: utf-8 -*-
# Author: 小狼狗

'''
================================================================
        LLE(Locally_linear_embedding)降维算法
================================================================
Input X: D by N matrix consisting of N data items in D dimensions.
Output Y: d by N matrix consisting of d < D dimensional embedding coordinates for the input points.


Find neighbours in X space [b,c].
for i=1:N
  compute the distance from Xi to every other point Xj
  find the K smallest distances
  assign the corresponding points to be neighbours of Xi
end
Solve for reconstruction weights W.
for i=1:N
  create matrix Z consisting of all neighbours of Xi [d] //把xi的k个近邻排成一个D*k的矩阵Z
  subtract Xi from every column of Z   //Z的每一列都减去xi
  compute the local covariance C=Z'*Z [e]   //C=Z.T*Z
  solve linear system C*w = 1 for w [f]     //求解w使C*w = 1
  set Wij=0 if j is not a neighbor of i
  set the remaining elements in the ith row of W equal to w/sum(w);
end
Compute embedding coordinates Y using weights W.
create sparse matrix M = (I-W)'*(I-W)
find bottom d+1 eigenvectors of M
  (corresponding to the d+1 smallest eigenvalues)
set the qth ROW of Y to be the q+1 smallest eigenvector
  (discard the bottom eigenvector [1,1,1,1...] with eigenvalue zero)
Notes

[a] Notation
    Xi and Yi denote the ith column of X and Y
      (in other words the data and embedding coordinates of the ith point)
    M' denotes the transpose of matrix M
    * denotes matrix multiplication
      (e.g. M'*M is the matrix product of M left multiplied by its transpose)
    I is the identity matrix
    1 is a column vector of all ones

[b] This can be done in a variety of ways, for example above we compute
    the K nearest neighbours using Euclidean distance.
    Other methods such as epsilon-ball include all points within a
    certain radius or more sophisticated domain specific and/or
    adaptive local distance metrics.

[c] Even for simple neighbourhood rules like K-NN or epsilon-ball
    using Euclidean distance, there are highly efficient techniques
    for computing the neighbours of every point, such as KD trees.

[d] Z consists of all columns of X corresponding to
    the neighbours of Xi but not Xi itself

[e] If K>D, the local covariance will not be full rank, and it should be
    regularized by seting C=C+eps*I where I is the identity matrix and
    eps is a small constant of order 1e-3*trace(C).
    This ensures that the system to be solved in step 2 has a unique solution.

[f] 1 denotes a column vector of all ones
'''
#===========================================================================
import numpy as np


def lle(X, K, d):
    X = X.T
    D,N = X.shape

    # STEP1: COMPUT PAIRWISE DISTANCES & FIND NEIGHBORS
    print(1, '-->Finding %d nearest neighbours.\n' % K)
    X2 = sum(X ** 2, 0)
    X2 = np.asmatrix(X2)

    #计算disatnce矩阵，即distance_ij表示第i个数据和第j个数据的距离，diatance是一个对称矩阵，对角线元素为0
    distance = np.tile(X2, (N, 1)) + np.tile(X2.T,(1,N)) - 2 * np.dot(X.T, X)
    index = np.argsort(distance, axis=0)
    #找出每个样本的k近邻，组成neighborhood矩阵
    neighborhood = index[1:K+1,:]
    #print(neighborhood)

    # STEP2: SOLVEFOR RECONSTRUCTION WEIGHTS

    print(2,'-->Solving for reconstruction weights.\n')
    if K>D:
        tol = 1e-3
    else:
        tol = 0
    W = np.zeros((N, N))
    for i in range(N):
        X = np.asmatrix(X)
        Z = X[:, neighborhood[:, i]].reshape(D,K) - X[:, i] # 计算Z

        C = np.dot(Z.T, Z)  # 计算G=Z'*Z
        C = C + np.eye(K) * tol * np.trace(C)

        W[neighborhood[:,i], i] = np.linalg.solve(C,-np.ones((K, 1)))  # 解方程Gw=e  e=[1,1,1..,1]

        W[:, i] = W[:, i] / sum(W[:, i]);  # 解向量W归一化
    print("  -->Done.")

    # STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M = (I - W)'(I-W)

    print(3,'-->Computing embedding.\n')
    #M = (I-W)*(I-W)'
    M =np.dot((np.eye(N) - W), (np.eye(N) - W).T)
    # v[:,i] is the eigenvector corresponding to the eigenvalue u[i]
    u, v = np.linalg.eig(M)
    #取Y为M的d个非零最小特征值对应的特征向量，即先将特征值从小到大排序，第一个特征值几乎接近0，所以舍弃第一个特征值，取2~d+1个
    u_index = np.argsort(np.abs(u))
    Y = v[:,u_index[1:d+1]]/np.sqrt(u[u_index[1:d+1]])
    return Y
#
if __name__ == '__main__':
    X = np.array([[-20, -8],
                  [-10, -1],
                  [0, 0.01],
                  [10, 1],
                  [20, 8]])
    Y = lle(X, 2, 1)
    print(Y)



#--------------------------------------------------------------------------------
###===================sklearn实现==============================
#
import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D

#----------------------------------------------------------------------
# Locally linear embedding of the swiss roll(or s_curve)

from sklearn import manifold, datasets
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)
#print(X.shape)     #1500*3
#print(color[1:5])

print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=15,
                                                 n_components=2)
print("Done. Reconstruction error: %g" % err)
#X_r = lle(X, K=13, d=2)

#----------------------------------------------------------------------
# Plot result

fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")
ax = fig.add_subplot(212)
ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()





























