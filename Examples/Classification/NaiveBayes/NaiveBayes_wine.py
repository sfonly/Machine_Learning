# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:29:09 2019

贝叶斯：GaussianNB

@author: sf_on
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from math import sqrt
from collections import Counter
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


def multipl(x, y):
    '''
    计算：x,y 两个向量的内积
        注意 x 和 y 一定是等行数的
    Paramters：
        x:       向量 x
        y:       向量 y
    Return：
        sumofab  两个向量的内积
    '''
    sumofab = 0.0
    for i in range(len(x)):
        temp = x[i] * y[i]
        sumofab = sumofab + temp
    return sumofab

def corrcoef(x,y):
    '''
    计算：x,y 两个向量的皮尔逊相关系数
        注意 x 和 y 一定是等行数的
    Paramters：
        x:       向量 x
        y:       向量 y
    Return：
        up/den  两个向量的皮尔逊相关系数
    '''
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sumofxy = multipl(x,y)
    x2 = []
    y2 = []
    
    for i in x:
        x2.append(pow(i,2))
    for j in y:
        y2.append(pow(j,2))
        
    sumofx2 = sum(x2)
    sumofy2 = sum(y2)
    up = sumofxy - (float(sum_x)*float(sum_y))/n
    den = sqrt((sumofx2 - float(pow(sum_x,2))/n)*(sumofy2 - float(pow(sum_y,2))/n))
    return up/den

def pearson(x,y):
    '''
    计算：特征矩阵 x 和类标号 y 之间的相关系数
        注意 x 和 y 一定是等行数的
    Paramters：
        x:       特征矩阵 x
        y:       向量 y
    Return：
        pearson_x  x特征矩阵和类标号y之间的相关系数
    '''
    pearson = []
    for col in x.columns:
        pearson.append(abs(corrcoef(x[col].values,y.values)))
    pearson_x = pd.DataFrame({'columns':x.columns,'corr_value':pearson})
    pearson_x = pearson_x.sort_values(by='corr_value',ascending=False)
    return pearson_x

def show_boxplot(data,cols,label):
    '''
    查看数据盒须图
    Paramters：
        data:       原始数据集
        cols:       特征
        label：     类标号
    '''
    for col in cols:
        sns.boxplot(x = data[label], y = data[col])
        sns.stripplot(x = data[label], y = data[col], jitter=True, edgecolor="gray")
        plt.show()


def find_outlier(data,features,label,n):
    '''
    寻找异常点
    Paramters：
        data:                数据集
        features：           特征
        label：              类标号
        n:                   设置有n个异常的特征时，认为是异常点
    Return：
        multiple_outliers：  异常值的list
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
    multiple_outliers=list(k for k,v in outlier_indices.items() if v >= n)
    return multiple_outliers

def show_corr(data):
    '''
    相关系数图热力图
    Paramters：
        data:       数据集
    '''
    cor = data.corr() 
    plt.figure(figsize=(12,10))
    sns.heatmap(cor)  
    plt.show()

def scatter_features_matrics(x,y,label):
    '''
    将连续特征进行矩阵显示
    Paramters：
        x:           输入特征空间
        y:           预测空间
        label：      类标号
    '''
    yl = copy.deepcopy(y)
    yl[label] = yl[label].replace([1,2,3],['b','g','r']) 
    pd.plotting.scatter_matrix(x, alpha=0.7, c=yl[label],figsize=(20,20), diagonal='hist')
    plt.show()

def classification(x,y):
    '''
    利用高斯贝叶斯对数据集进行分类测试
    Paramters：
        x:           输入特征空间
        y:           预测空间
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,random_state=6)
    model = GaussianNB()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    print('accuracy score:', model.score(x_test,y_test))
    print(metrics.classification_report(expected, predicted)) 
    print(confusion_matrix(expected, predicted, sample_weight=None))

def output_csv(x,y):
    '''
    导出训练文件和测试文件
    Paramters：
        x:       特征矩阵
        y:       类标号
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,random_state=6)
    Traindata = pd.merge(x_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/wine_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(x_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/wine_test.csv',index = False,encoding = 'utf-8')



if __name__ == '__main__':
    
    # 导入数据源，原始数据不含表头，需要重新创建
    columns = ['category','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
             'Total phenols','Flavanoid','Nonflavanoid phenols','Proanthocyanins',
             'Color intensity','Hue','OD280/OD315 of diluted wines' ,'Proline']
    data = pd.read_csv(open('./wine.csv'),
                       header = None, names = columns)
    
    # 查看数据集基本状况
    print(data.shape)
    print(data.head())
    print(data.info())
    print(data['category'].describe())
    print(data['category'].value_counts())
    
    # 特征的list
    features_cols = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
                     'Total phenols','Flavanoid','Nonflavanoid phenols','Proanthocyanins',
                     'Color intensity','Hue','OD280/OD315 of diluted wines' ,'Proline']
    
    # 处理数据集中的异常值
    show_boxplot(data,features_cols,'category')
    outlier = find_outlier(data,features_cols,'category',3)
    data = data.drop(outlier,axis = 0)
    data.reset_index(drop=True, inplace=True)
    
    # 装载特征和类标号
    x = data[features_cols]
    y = data[['category']]

    # 查看特征与类标号，特征与特征间的相关性
    pearson_xy = pearson(x,y)
    corr = x.corr()
    show_corr(x)
    print(pearson_xy)
    x = x.drop(['Flavanoid','Ash'],axis = 1)
    
    scatter_features_matrics(x,y,'category')    

    classification(x,y)    
    # output_csv(x,y)

