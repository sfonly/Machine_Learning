
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:22:59 2019

@author: sf_on
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def tree_adaboost_show(x,y):
    
    x_min = x[:,0].min() - 1
    x_max = x[:,0].max() + 1
    y_min = x[:,1].min() - 1
    y_max = x[:,1].max() + 1
    
    xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1), np.arange(y_min,y_max,0.1))
    f,axarr = plt.subplots(1,2,sharex="col",sharey="row",figsize=(12,5))

    for idx,clf,tt in zip([0,1],[tree,adaboost],["决策树","AdaBoost"]):
        clf.fit(x,y)
        Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx,yy,Z,alpha=0.3)
        #绘制第一类训练集的点
        axarr[idx].scatter(x[y == 1, 0],x[y == 1, 1],c="blue",marker="^")
        #绘制第二类训练集的点
        axarr[idx].scatter(x[y == 2, 0],x[y == 2, 1],c="red",marker="o")
        #绘制第三类训练集的点
        axarr[idx].scatter(x[y == 3, 0],x[y == 3, 1],c="green",marker="*")
        axarr[idx].set_title(tt)
        
    axarr[0].set_ylabel("酒精度(Alcohol)",fontsize=12)
    plt.text(10.0,-0.8,s="色度(Hue)",ha="center",va="center",fontsize=12)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.show()



if __name__ == '__main__':
    
    columns=['category','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
             'Flavanoid','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue',
             'OD280/OD315 of diluted wines' ,'Proline']
    data = pd.read_csv(open('C://Users/sf_on/Desktop/数据挖掘应用分析实验手册/贝叶斯/wine.csv'),
                       header = None, names = columns)

    X = data[['Alcohol','Hue']].values
    label_y = data['category'].values
 
    train_x, test_x, train_y, test_y = train_test_split(X, label_y, test_size=0.3, random_state=1)
    
    tree = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
    tree.fit(train_x,train_y)
    tree_train_predict = tree.predict(train_x)
    tree_test_predict = tree.predict(test_x)
     
    print('tree train accuracy:', accuracy_score(train_y, tree_train_predict))
    print('tree test accuracy:', accuracy_score(test_y, tree_test_predict))
 
    adaboost = AdaBoostClassifier(base_estimator= tree, n_estimators = 900, learning_rate=0.05, algorithm = 'SAMME')
    adaboost.fit(train_x,train_y)
    adaboost_train_predict = adaboost.predict(train_x)
    adaboost_test_predict = adaboost.predict(test_x)
 
    print('adaboost train accuracy:', accuracy_score(train_y, adaboost_train_predict))
    print('adaboost test accuracy:', accuracy_score(test_y, adaboost_test_predict))
 
    tree_adaboost_show(train_x, train_y)
