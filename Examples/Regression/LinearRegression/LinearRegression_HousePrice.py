# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
data = pd.read_csv('./HousePrice.csv')
# 读入数据

#data.isnull().any().sum()

#pd.plotting.scatter_matrix(data, alpha=0.5, figsize=(10,10), diagonal='kde')

corr = data.corr()

x = data[['CRIM','CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
          'TAX','PTRATIO','B','LSTAT']]
y = data[['MEDV']]
# 设置变量和类标号

# 找到和y相关性最高的3个变量
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
SelectKBest = SelectKBest(f_regression, k=3)
bestFeature = SelectKBest.fit_transform(x,y)
SelectKBest.get_support()
x.columns[SelectKBest.get_support()]

features = data[x.columns[SelectKBest.get_support()]]
pd.plotting.scatter_matrix(features, c='b',alpha=0.7, figsize=(12,12), diagonal='hist')
plt.subplots(figsize=(8,8))
plt.scatter(features['LSTAT'],y,s = 20)
plt.title('LSTAT')  
plt.show()

# 获取y中MEDV等于50的index，并存储在list中
i_ = y[y.MEDV == 50].index.tolist()
print(i_)
y.drop(i_,axis=0,inplace=True) #删除异常值
features.drop(i_,axis=0,inplace=True) #删除异常值

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for feature in features.columns:
    features['标准化'+feature] = scaler.fit_transform(features[[feature]])
pd.plotting.scatter_matrix(
        features[['标准化RM','标准化PTRATIO', '标准化LSTAT']], 
        alpha=0.7, figsize=(8,8), diagonal='hist')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        features[['标准化RM','标准化PTRATIO', '标准化LSTAT']],y, test_size=0.3,random_state=33)

# =============================================================================
# 生成训练集csv文件
# Traindata = pd.DataFrame(columns=('INDUS','RM', 'TAX','PTRATIO','LSTAT','MEDV'))
# Traindata['INDUS'] = x_train['标准化INDUS']
# Traindata['RM'] = x_train['标准化RM']
# Traindata['TAX'] = x_train['标准化TAX']
# Traindata['PTRATIO'] = x_train['标准化PTRATIO']
# Traindata['LSTAT'] = x_train['标准化LSTAT']
# Traindata['MEDV'] = y_train['MEDV']
# Traindata.to_csv('C://Users/sf_on/Desktop/houseprice_train.csv',index = False,encoding = 'utf-8')
# =============================================================================

# =============================================================================
# Testdata = pd.DataFrame(columns=('INDUS','RM', 'TAX','PTRATIO','LSTAT','MEDV'))
# 生成测试集csv文件
# Testdata['INDUS'] = x_test['标准化INDUS']
# Testdata['RM'] = x_test['标准化RM']
# Testdata['TAX'] = x_test['标准化TAX']
# Testdata['PTRATIO'] = x_test['标准化PTRATIO']
# Testdata['LSTAT'] = x_test['标准化LSTAT']
# Testdata['MEDV'] = y_test['MEDV']
# Testdata.to_csv('C://Users/sf_on/Desktop/houseprice_test.csv',index = False,encoding = 'utf-8')
# =============================================================================
    
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
print(lr.coef_)
print(lr.intercept_)

y_predict=lr.predict(x_test)
lr.score(x_train,y_train)
lr.score(x_test,y_test)

ROCdata = pd.DataFrame(columns=('MEDV','P_MEDV'))
ROCdata['MEDV'] = y_test['MEDV']
ROCdata['P_MEDV'] = y_predict

from sklearn.metrics import mean_squared_error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predict))

ROCdata = ROCdata.sort_index(by='P_MEDV',axis=0,ascending=True)
plt.subplots(figsize=(14,7))
plt.plot(range(len(ROCdata)),ROCdata['MEDV'],'green',linewidth=2.5,label="real_data")
plt.plot(range(len(ROCdata)),ROCdata['P_MEDV'],'red',linewidth=2.5,label="pre_data")
plt.legend(loc=2)
plt.show()

