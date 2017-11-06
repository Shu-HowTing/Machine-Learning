# A = [[1,2,4,5],
#     [2,3,4,5],
#     [3,4,5,6]]
# a=[x[2] for x in A]
# print(a)
# d1 = {}
# for i in range(3):
#     d1.setdefault(1, []).append(A[i])
# print(d1)
# lst = [2,4,5]
# print(sum(lst))
# d = {}
# for i in range(len(A)):
#     if i in d:
#         d[i].append(A[i])
#     else:
#         d[i]=A[i]
# print(d)
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# data = np.array(np.random.randint(-100,100,24).reshape(6,4)).astype(float)
# x = MinMaxScaler((0,1), copy=True)
# y = x.fit_transform(data)
# #X = x.transform(data)
# print(data)
# print(y)
# #print(new_data)
import numpy as np
import csv
<<<<<<< HEAD
# f = lambda x_y: x_y[0] + x_y[1]
# print(f((3,4)))
#
# A = [[1,2,3],[2,3,4],[3,4,5]]
# print(np.argsort(A,axis=0))
#
# print(np.log(np.e))
# a = 8
# b = a
# a = 0
# print(b)
#
#
# for i in range(1,4):
#     print(i)
#
# a = 4
# b = np.copy(a)
# a = 5
# print(b)
# import matplotlib.pyplot as plt
# a = [1,2,3,4]
# b = [2,3,5,6]
# x = [0,1,2,3,4,5,6]
# y = [0.3,0.4,2,5,3,4.5,4]
#
# X = map(int, y)
# for i in X:
#     print(i)
#
# #导入txt文件
# X = np.loadtxt('HorseColicTest.txt')
# print(X.shape)
# print(type(X[1][1]))

A = np.array([[1,2,3],
             [2,3,4],
             [3,4,5]])
a = [1,2,5]
print(a-A)
print(np.multiply(A,A))
print(np.dot(A, A))
l = np.array([1,23,45,6,7,8,9,9,0,6,5,43])
print(np.array_split(l,4))
=======
f = lambda x_y: x_y[0] + x_y[1]
print(f((3,4)))

A = [[1,2,3],[2,3,4],[3,4,5]]
print(np.argsort(A,axis=0))

print(np.log(np.e))
a = 8
b = a
a = 0
print(b)


for i in range(1,4):
    print(i)

a = 4
b = np.copy(a)
a = 5
print(b)
import matplotlib.pyplot as plt
a = [1,2,3,4]
b = [2,3,5,6]
x = [0,1,2,3,4,5,6]
y = [0.3,0.4,2,5,3,4.5,4]

X = map(int, y)
for i in X:
    print(i)

#导入txt文件
X = np.loadtxt('HorseColicTest.txt')
print(X.shape)
print(type(X[1][1]))
>>>>>>> d342b1271f8f34211a4b1da7b8d64ed320fe6cf1
print("hello world")
