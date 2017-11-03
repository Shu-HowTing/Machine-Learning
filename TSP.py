# -*- coding: utf-8 -*-
# Author: 小狼狗

'''
===========================================================
            模拟退火算法：TSP问题
===========================================================

模拟退火算法:
     参数：T代表原始温度，cool代表冷却率，step代表每次选择临近解的变化范围
     原理：退火算法以一个问题的随机解开始，用一个变量表示温度，这一温度开始时非常高，而后逐步降低
          在每一次迭代期间，算啊会随机选中题解中的某个数字，然后朝某个方向变化。如果新的成本值更
          低，则新的题解将会变成当前题解，这与爬山法类似。不过，如果成本值更高的话，则新的题解仍
          有可能成为当前题解，这是避免局部极小值问题的一种尝试。
     注意：算法总会接受一个更优的解，而且在退火的开始阶段会接受较差的解，随着退火的不断进行，算法
          原来越不能接受较差的解，直到最后，它只能接受更优的解。
     算法接受较差解的概率 P = exp[-(highcost-lowcost)/temperature]
'''
import numpy as np
import matplotlib.pyplot as plt

coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                        [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
                        [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
                        [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
                        [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
                        [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
                        [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
                        [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
                        [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
                        [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
                        [1340.0, 725.0], [1740.0, 245.0]])

coordinates.shape  # (52, 2)


def getdistmat(coordinates):
    num = coordinates.shape[0] #行数 num = 52
    distmat = np.zeros((52, 52))
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(coordinates[i] - coordinates[j]) #算距离
    return distmat


getdistmat(coordinates).shape  # (52, 52)


def init_para():
    alpha = 0.95   #温度t的冷却速度
    t = [10, 100]    #t[1] = 100 起始温度 ，t[0]=1 终止温度
    markovlen = 10000
    return alpha, t, markovlen


num = coordinates.shape[0] #num =52
distmat = getdistmat(coordinates) #52*52

solutionnew = np.arange(num)  # [0,1,2,3,4 .....51]
#valuenew = np.max
valuenew = 0

solutioncurrent = solutionnew.copy()
#valuecurrent = np.max
valuecurrent = np.sum(distmat[0,:])

solutionbest = solutionnew.copy()
#valuebest = np.max
valuebest = np.sum(distmat[0,:])

alpha, t, markovlen = init_para()
T = t[1] # T=100

result = []  # 记录迭代过程中的最优解

while T > t[0]:
    for i in np.arange(markovlen):
        # print t, i
        # 下面的两交换和三角换是两种扰动方式，用于产生新解
        if np.random.rand() > 0.5:  # 两交换
            # np.random.rand()产生[0, 1)区间的均匀随机数
            while True:  # 产生两个不同的随机数
                #ceil()方法返回x的值上限 - 不小于x的最小整数。
                loc1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                loc2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                if loc1 != loc2:
                    break
            solutionnew[loc1], solutionnew[loc2] = solutionnew[loc2], solutionnew[loc1]
        else:  # 三交换
            while True:
                loc1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                loc2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                loc3 = np.int(np.ceil(np.random.rand() * (num - 1)))

                if ((loc1 != loc2) & (loc2 != loc3) & (loc1 != loc3)):
                    break
            # 下面的三个判断语句使得loc1<loc2<loc3
            if loc1 > loc2:
                loc1, loc2 = loc2, loc1
            if loc2 > loc3:
                loc2, loc3 = loc3, loc2
            if loc1 > loc2:
                loc1, loc2 = loc2, loc1

             # 下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
            tmp_list = solutionnew[loc1: loc2].copy()
            solutionnew[loc1: loc1 + loc3 - loc2 + 1 ] = solutionnew[loc2: loc3 + 1].copy()
            solutionnew[loc3 - loc2 + 1 + loc1: loc3 + 1] = tmp_list.copy()

        valuenew = 0
        for i in range(num - 1):
            valuenew += distmat[solutionnew[i]][solutionnew[i + 1]]
        valuenew += distmat[solutionnew[0]][solutionnew[51]]

        if valuenew < valuecurrent:  # 接受该解
            # 更新solutioncurrent 和solutionbest
            valuecurrent = valuenew
            solutioncurrent = solutionnew.copy()

            if valuenew < valuebest:
                valuebest = valuenew
                solutionbest = solutionnew.copy()
        else:  # 按一定的概率接受该解
            if np.random.rand() < np.exp(-(valuenew - valuecurrent) / T):
                valuecurrent = valuenew
                solutioncurrent = solutionnew.copy()
            else:
                solutionnew = solutioncurrent.copy()

    T = alpha * T
    result.append(valuebest)
    print(T)
      # 程序运行时间较长，打印T来监视程序进展速度

plt.plot(np.array(result))
plt.ylabel("bestvalue")
plt.xlabel("n")
plt.show()
