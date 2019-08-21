# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:56:50 2019

Kmeans

@author: sf_on
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

def show_boxplot(data,col,label):
    '''
    查看数据盒须图
    Paramters：
        data:       原始数据集
        cols:       特征
        label：     类标号
    '''
    plt.figure(figsize = (16,8))
    plt.rcParams['font.sans-serif']=['SimHei']
    sns.boxplot(x = data[label], y = data[col])
    plt.show()

def find_outlier(data, col,label):
    '''
    根据组别寻找某个特征的异常点
    Paramters：
        data:               数据集
        col                 特征
        label:              类标号
    Return：
        outlier_list_col:  异常值的list
    '''
    grouped = data.groupby(data[label])
    outlier_index = []
    all_province = []
    for name, group in grouped:
        Q1 = np.percentile(group[col],25)
        Q3 = np.percentile(group[col],75)
        IQR = Q3 - Q1
        outlier_step = 3 * IQR
        tmp = group[(group[col] < Q1 - outlier_step)|(group[col] > Q3 + outlier_step)].index.tolist()
        outlier_index.extend(tmp)
        all_province.append([name, Q1, Q3, IQR])
    return outlier_index, all_province

def print_outlier(data, outlier_index, all_province, name):
    '''
    输出异常点的信息
    Paramters：
        data:               数据集
        outlier_index       异常点的index
        all_province:       所有省分的 Q1 Q3 IQR等信息
        name：              经纬度
    '''
    print(name,'\n')
    for ol in outlier_index:
        for row in all_province:
            if data.iloc[ol,0] == row[0]:
                print('province：',row[0])
                print('Q1',row[1],', Q3',row[2],', IQR', float('%.2f' % row[3]))
                if name == 'longitude':
                    print('index:',ol,', outlier:', data.iloc[ol,[0,1,2]].tolist())
                if name == 'latitude':
                    print('index:',ol,', outlier:', data.iloc[ol,[0,1,3]].tolist())
                print('')

def outlier_position(data,label):
    '''
    确定异常点
    Paramters：
        data:           数据集
        label           类别
    Return：
        outlier:        异常的list
    '''
    outlier = []
    outlier_longitude_index,longitude_all  = find_outlier(data, 'longitude', label)
    outlier_latitude_index, latitude_all  = find_outlier(data, 'latitude', label)
    outlier.extend(outlier_longitude_index)
    outlier.extend(outlier_latitude_index)
    
    print_outlier(data, outlier_longitude_index, longitude_all, 'longitude')
    print_outlier(data, outlier_latitude_index, latitude_all, 'latitude')
    return outlier

def city_show(data, n_cluster, cls):
    '''
    绘制城市散点地图
    Paramters：
        data:            数据
        n_cluster:       簇团
        cls:             聚类模型
    '''
    X = []
    for index, row in data.iterrows():
        X.append([float(row['longitude']), float(row['latitude'])])
    X = np.array(X)
    
    # markers = ['*','o','+','s','v','1','h','X']
    markers = ['*','o','+','s','v']
    colors = ['b','c','y','m','g']
    
    plt.figure(figsize=(12,8))
    for i in range(n_cluster):
        # members是布尔数组
        members = cls.labels_ == i
        # 画与menbers数组中匹配的点
        plt.scatter(X[members,0], X[members,1], s=20, marker = markers[i],
                    c = colors[i], alpha=0.5)
    plt.title('China City Clustering')
    plt.show()


if __name__ == '__main__':
    
    columns = ['province','city','longitude','latitude']
    dataframe = pd.read_csv(open('./city.csv',
                            encoding='UTF-8'),encoding='UTF-8', header = None, names = columns)

    # 数据去重
    data = dataframe.drop_duplicates(keep='first').reset_index(drop=True)
    print(data.describe())
    print(data.info())

    # 查看各省份经纬度盒须图
    show_boxplot(data,'longitude','province')
    show_boxplot(data,'latitude' ,'province')
    
    # 去除异常值
    outlier = outlier_position(data,'province')
    outlier.remove(1537)
    outlier.remove(1538)
    data = data.drop(outlier, axis = 0)
    data = data.reset_index(drop=True)
    
    # 利用kmeans进行聚类
    n_cluster = 5
    cls = KMeans(n_cluster).fit(data.loc[:,['longitude','latitude']])
    
    # 用scatter方法绘制二维的聚类效果图
    city_show(data, n_cluster, cls)


