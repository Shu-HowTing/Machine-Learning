# -*- coding: utf-8 -*-
# Author: 小狼狗
'''
==================================================================
    An implementation of matrix factorization(推荐系统)
==================================================================
'''
import numpy

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
###############################################################################
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:    #从实际角度考虑，认为只有大于0的数据才有意义
                    e_ij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * e_ij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * e_ij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R) #N*M

    N = len(R)      # 5
    M = len(R[0])   # 4
    K = 2
# random.rand
# Create an array of the given shape
# and populate it with random samples from a uniform distribution over [0, 1).
    P = numpy.random.rand(N,K)    # 5*2
    Q = numpy.random.rand(M,K)    # 4*2

    nP, nQ = matrix_factorization(R, P, Q, K)
    print("P:{}".format(nP))
    print("Q:{}".format(nQ))
    print(numpy.dot(nP,nQ.T))