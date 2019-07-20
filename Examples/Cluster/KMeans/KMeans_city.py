# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:56:50 2019

@author: sf_on
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

columns = ['province','city','longitude','latitude']
dataframe = pd.read_csv(open('./city_cluster.csv',
                        encoding='UTF-8'),encoding='UTF-8', header = None, names = columns)

data = dataframe.drop_duplicates(keep='first').reset_index(drop=True)
print(data.describe())

plt.figure(figsize = (16,8))
sns.boxplot(x = data['province'], y = data['longitude'])
plt.show()

def find_outlier(data, col):
    grouped = data.groupby(data['province'])
    outlier_list_col = []
    for name, group in grouped:
        Q1 = np.percentile(group[col],25)
        Q3 = np.percentile(group[col],75)
        IQR = Q3 - Q1
        outlier_step = 3 * IQR
        tmp = group[(group[col] < Q1 - outlier_step)|(group[col] > Q3 + outlier_step)].index.tolist()
        outlier_list_col.extend(tmp)
    return outlier_list_col
outlier_longitude = find_outlier(data, 'longitude')
outlier_latitude = find_outlier(data, 'latitude')
outlier = []
outlier.extend(outlier_longitude)
outlier.extend(outlier_latitude)
for ol in outlier:
    print(data.iloc[ol,:].tolist())
outlier.remove(1537)
outlier.remove(1538)
data = data.drop(outlier, axis = 0)
data = data.reset_index(drop=True)

X = []
for index, row in data.iterrows():
    X.append([float(row['longitude']), float(row['latitude'])])
X = np.array(X)
n_cluster = 5

cls = KMeans(n_cluster).fit(X)
# markers = ['*','o','+','s','v','1','h','X']
markers = ['*','o','+','s','v']
colors = ['b','c','y','m','g']

def city_show(X, n_cluster):
    plt.figure(figsize=(12,8))
    for i in range(n_cluster):
        # members是布尔数组
        members = cls.labels_ == i
        # 画与menbers数组中匹配的点
        plt.scatter(X[members,0],X[members,1],s=20,marker=markers[i],c=colors[i],alpha=0.5)
    plt.title('China')
    plt.show()
city_show(X, n_cluster)


