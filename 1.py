'''
=============================================================
                    模式识别作业
=============================================================
'''
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import grid_search
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
lst =[]
with open('data.csv','r') as f:
    data = csv.reader(f)
    for row in data:
        lst.append(row)
lst =np.reshape(lst,[150,3])
X = lst[:,1:3]
y = lst[:,0]
#print(X,y)
X_train, X_test, y_train, y_test = train_test_split(
                           X,y,test_size=0.2)

parameter = {'kernel':("linear","rbf"), "C":[1, 10, 100]}
svm = SVC()

clf = grid_search.GridSearchCV(svm, parameter)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print (clf.best_score_,clf.best_params_)
print('accurancy:',clf.score(X_test, y_test))
#plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(X[:, 0],X[:,1] ,c=y)
plt.xlabel('N', fontsize=14)
plt.ylabel('PRT', fontsize=14)
plt.show()



