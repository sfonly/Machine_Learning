# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:29:09 2019

@author: sf_on
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# 计算：a,b中每个元素乘积的连加
def multipl(a,b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i]*b[i]
        sumofab = sumofab + temp
    return sumofab

#计算：x,y的皮尔逊相关系数
def corrcoef(x,y):
    n = len(x)
    #求和
    sum_x = sum(x)
    sum_y = sum(y)
    #求乘积和
    sumofxy = multipl(x,y)
    #求平方和
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

#计算：欧式距离,计算多维度的向量，vector1和vector2的维度必须相同
def distance(vector1,vector2):
    d = 0;
    for a,b in zip(vector1,vector2):
        d = pow(a-b,2) + d
    return sqrt(d)

# 皮尔逊相关系数计算
def pearson(x,y):
    pearson = []
    for col in x.columns:
        pearson.append(abs(corrcoef(x[col].values,y.values)))
    pearson_x = pd.DataFrame({'columns':x.columns,'corr_value':pearson})
    pearson_x = pearson_x.sort_values(by='corr_value',ascending=False)
    print(pearson_x)
    return pearson_x

# 划分训练集和测试集
def selection(x,y):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.2,random_state=6)
    return x_train, x_test, y_train, y_test

# 利用某方法对数据进行分类并测试
def function_classification(x,y,function):
    # 切分数据
    x_train, x_test, y_train, y_test = selection(x,y)
    model = function()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    print(model.score(x_test,y_test))
    print(metrics.classification_report(expected, predicted)) 
    print(confusion_matrix(expected, predicted, sample_weight=None))

# 导出训练文件和测试文件
def output_csv(x,y):
    x_train, x_test, y_train, y_test = selection(x,y)    
    Traindata = pd.merge(x_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/wine_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(x_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/wine_test.csv',index = False,encoding = 'utf-8')


# 导入数据源，原始数据不含表头，需要重新创建
columns=['category','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
         'Flavanoid','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue',
         'OD280/OD315 of diluted wines' ,'Proline']
data = pd.read_csv(open('./wine.csv'),
                   header = None, names = columns)

data.shape
data.head()
data.info()

data['category'].describe()
data['category'].value_counts()

# 查看数据盒须图
for i in data.iloc[:,1:14].columns:
    sns.boxplot(x = data['category'], y = data[i])
    sns.stripplot(x = data['category'], y = data[i], jitter=True, edgecolor="gray")
    plt.show()
    
x = data[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols',
         'Flavanoid','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue',
         'OD280/OD315 of diluted wines' ,'Proline']]
y = data[['category']]

pearson_xy = pearson(x,y)

pd.plotting.scatter_matrix(x, alpha=0.7, figsize=(24,24), diagonal='hist')

cor1 = x.corr()
sns.heatmap(cor1)  
plt.show()

feature = data[['Alcohol','Malic acid','Alcalinity of ash','Magnesium',
                'Flavanoid','Nonflavanoid phenols','Proanthocyanins',
                'Color intensity','Hue','Proline']]

function_classification(feature,y,GaussianNB)
function_classification(feature,y,LogisticRegression)

output_csv(x,y)

