# -*- coding: utf-8 -*-
# Author: 小狼狗
'''
======================================================
                 sklearn实现GDBT
======================================================
'''

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#数据集
X = load_boston().data
Y = load_boston().target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

'''
参数：
    loss:损失函数，对于Regressor可以用'ls','lad','huber', 'quantile'。
    learning_rate: 学习率/步长。
    n_estimators: 迭代次数，和learning_rate存在trade-off关系。
    criterion: 衡量分裂质量的公式，一般默认即可。
    subsample: 样本采样比例。
    max_features: 最大特征数或比例。

    决策树相关参数包括max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                                max_leaf_nodes, min_impurity_split, 多数用来设定决策树分裂停止条件。

'''
reg_model = GradientBoostingRegressor(
                                        loss = "ls",
                                        learning_rate=0.02,
                                        n_estimators=200,
                                        subsample=0.8,
                                        max_features=0.8,
                                        max_depth=3,
                                        verbose=2
                                    )

reg_model.fit(X_train, y_train)

prediction_train = reg_model.predict(X_train)
rmse_train = mean_squared_error(y_train, prediction_train)

prediction_test = reg_model.predict(X_test)
rmse_test = mean_squared_error(y_test, prediction_test)

print("Training RMSE : %.3f"% rmse_train)
print("Testing RMSE :%.3f"% rmse_test)