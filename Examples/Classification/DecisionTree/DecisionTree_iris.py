# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:29:52 2019

决策树： DecisionTree

@author: sf_on
"""

import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree 

def show_box(data, features, label): 
    '''
    查看数据盒须图
    Paramters：
        data:       原始数据集
        cols:       特征
        label：     类标号
    '''
    for feature in features:
        sns.boxplot(x = label, y = feature, data = data)
        plt.show()

def find_outlier(data,features,label):    
    '''
    寻找异常点
    Paramters：
        data:                数据集
        features：           特征
        label：              类标号
    Return：
        multiple_outliers：  具有3个异常特征时的特征值index的list
    '''
    grouped = data.groupby(data[label])
    outlier_list = []
    for col in features:
        for name,group in grouped:
            Q1 = np.percentile(group[col], 25)
            Q3 = np.percentile(group[col], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            tmp = group[(group[col] < Q1 - outlier_step)|(group[col] > Q3 + outlier_step)].index.tolist()
            outlier_list.extend(tmp)
    outlier_indices = Counter(outlier_list)
    multiple_outliers=list(k for k,v in outlier_indices.items() if v > 2)
    return multiple_outliers

def feature_matrix_show(data):
    '''
    实现散点矩阵图
    Paramters：
        data:       原始数据集
    '''
    grouped = data.groupby(data['class'])
    p = []
    for col in features:
        temp = []
        for name,group in grouped:
            temp.append(group[col].tolist())
        p.append(temp)
    n = 0
    length = int(len(features))
    cube = 1
    plt.figure(figsize=(12,10),dpi=80)
    #创建第一个画板
    plt.figure(1)
    for i in range(length):
        for j in range(length):
            plt.subplot(length,length,cube)
            if i == j:
                plt.hist(p[n],stacked=True,color=['blueviolet','b','g'],align="right")
                n += 1
                cube += 1
            else:
                plt.scatter(data.iloc[:,j],data.iloc[:,i],alpha=0.6, s=5,c=data.iloc[:,4])
                cube += 1
    plt.tight_layout(pad=0)

def DecisionTree_classification(data, mode):
    '''
    利用决策树对其分类
    Paramters：
        data:       原始数据集
        cols:       特征
        label：     类标号
    '''
    # 切割数据并绘制模型，由于模型比较简单，就没有做剪枝和参数配置
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:4], data.iloc[:,4], test_size=0.3,random_state=32)
    
    clf = DecisionTreeClassifier(criterion = mode)
    clf.fit(X_train,y_train)
    
    print('clf.score:',clf.score(X_test,y_test))
    
    # 绘制决策树图，需安装有graphviz包
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=features,class_names=True, filled=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()



if __name__ == '__main__':
    
    columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
    data = pd.read_csv(open(./iris.csv'),header=None,names = columns)
    
    print(data.describe())
    print(data.isnull().sum())
    
    features = ['sepal_length','sepal_width','petal_length','petal_width']
    label = 'class'
    
    # 绘制盒须图，寻找异常点
    show_box(data, features,label)
    multiple_outliers = find_outlier(data,features,label)
    print('multiple_outliers',multiple_outliers)
    
    # 绘制散点柱状矩阵图
    data_show = data.replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],['blueviolet','blue','g']) 
    pd.plotting.scatter_matrix(data_show.iloc[:,:4], alpha=0.7, c=data_show.iloc[:,4],figsize=(12,12), diagonal='hist')
    # feature_matrix_show(data_show)

    DecisionTree_classification(data, 'entropy')
