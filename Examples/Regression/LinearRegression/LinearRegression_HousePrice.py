# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:43:26 2019

LinearRegression_HousePrice

@author: sf_on
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler,Normalizer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def show_col2label(data,col):
    '''
    查看列和类标号之间的关联关系
    Paramters：
        data:       数据集
        col:        列
    '''
    plt.subplots(figsize=(8,8))
    plt.scatter(data[col],y,s = 20)
    plt.title(col)  
    plt.show()

def show_line(y_test, y_predict):
    '''
    绘制查看预测曲线和真实值的偏差
    Paramters：
        y_test:         测试集的真实值
        y_predict:      测试集的预测值
    '''
    predata = pd.DataFrame(columns=('MEDV','P_MEDV'))
    predata['MEDV'] = y_test['MEDV']
    predata['P_MEDV'] = y_predict
    predata = predata.sort_index(by='P_MEDV',axis=0,ascending=True)
    plt.subplots(figsize=(14,7))
    plt.plot(range(len(predata)),predata['MEDV'],'green',linewidth=2.5,label="real_data")
    plt.plot(range(len(predata)),predata['P_MEDV'],'red',linewidth=2.5,label="pre_data")
    plt.legend(loc=2)
    plt.show()

def find_k_important_feature(x,y,k=5):
    '''
    查找y相关性最高的k个变量
    Paramters：
        y_test:         测试集的真实值
        y_predict:      测试集的预测值
    Return:
        important_features:     前 k 个重要的指标
    '''
    Kbest = SelectKBest(f_regression, k)
    Kbest.fit_transform(x,y)
    important_features = x.columns[Kbest.get_support()]
    return important_features

# 绘制相关性矩阵
def show_minmaxdata_matrix(data):
    '''
    查找y相关性最高的k个变量
    Paramters：
        data:       数据集
    '''
    pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(16,16), diagonal='hist')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.show()

def dropdata(x,y,i_): 
    '''
    同时drop特征与类标号
    Paramters：
        x:      特征
        y:      类标号
        i_:     要drop掉的行的index
    '''
    x.drop(i_,axis=0,inplace=True)
    y.drop(i_,axis=0,inplace=True)

def show_corr(data):
    '''
    查看相关性热力图
    Paramters：
        data:       数据集
    '''
    plt.figure(figsize=(10,8))
    g = sns.heatmap(data.corr(),annot=True,fmt = '.2f',cmap = 'coolwarm')
    g.set_xlabel('corr')
    plt.show()

def data_transformation(data, columns, Function):
    '''
    数据标准化 or 数据归一化方法封装
    Paramters：
        data:       数据集
        columns:    列
        Function:   方法
    Return:
        features_transformation:    转换数据的对象
    '''
    function = Function()
    features_transformation = pd.DataFrame(function.fit_transform(features), columns=columns)
    return features_transformation

def analysis(x,y):
    '''
    训练模型
    Paramters：
        x:          特征
        y:          类标号
    '''
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=32)
    
    lr = linear_model.LinearRegression()
    lr.fit(x_train,y_train)
    y_predict=lr.predict(x_test)
    
    print('lr_model coef:', lr.coef_)
    print('lr_model intercept:', lr.intercept_)
    print('lr_model train_score:', lr.score(x_train,y_train))
    print('lr_model test_score:', lr.score(x_test,y_test))
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))    
    
    show_line(y_test, y_predict)

if __name__ == '__main__':

    data = pd.read_csv(open('./HousePrice.csv'))
    
    # 数据探索
    print(data.isnull().sum())
    print(data.describe())
    print(data.MEDV.describe())
    
    x = data[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
              'DIS','RAD','TAX','PTRATIO','B','LSTAT']]
    y = data[['MEDV']]
    
    # 查看特征的相关性, drop 掉 RAD
    show_corr(x)
    x.drop('RAD', axis=1,inplace=True)
    
    # 寻找和类标号最相关的 k 个特征
    index = find_k_important_feature(x,y, 12)
    features = data[index]
    columns = features.columns.tolist()
    
    # 画出每个特征和类标号的关联性
    for col in index:   
        show_col2label(features,col)
    
    # 获取y中MEDV等于50的index，并存储在list中
    i_ = y[y.MEDV == 50].index.tolist()
    dropdata(features,y,i_)
        
    # 数据转化方法封装
    features_MinMaxScaler =  data_transformation(features, columns, MinMaxScaler)
    features_Normalizer   =  data_transformation(features, columns, Normalizer)
    
    # 归一化后
    print('This is MinMaxScaler Data: ')
    show_minmaxdata_matrix(features_MinMaxScaler)
    analysis(features_MinMaxScaler,y)
    
    # 标准化后
    # 画出每个特征和类标号的关联性
    print('This is Normalizer Data: ')
    for col in index:   
        show_col2label(features_Normalizer,col)
    show_minmaxdata_matrix(features_Normalizer)
    analysis(features_Normalizer,y)

