# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:39:17 2019

感知机：Perceptron

@author: sf_on
"""

import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class perceptron(object):
    
    def __init__(self):
        '''
        初始化感知机
        Parameters:
            learning_step：      学习速率
            max_iteration：      最大迭代次数
        '''
        self.learning_step = 0.01
        self.max_iteration = 1000 
        
    def fit(self, features, labels):
        '''
        训练感知机
        Parameters:
            features：       特征
            labels：         类标号
        结果：
            最后生成self对象的特征权重 W 向量
        '''
        
        self.w = [0.0] * (len(features[0])+1) # 加上w0
        correct_count = 0 # 被正确分类的次数，用于控制迭代
        
        while correct_count < self.max_iteration:
            index = random.randint(0, len(labels) -1) # 采用随机梯度下降，每次只更新一个样本点
            x = features[index].tolist()
            x.append(1.0) # 加上x0
            
            # 将问题转化为 y*(wx+b)=1 的问题，y=1或者-1
            y = 2 * labels[index] -1 
            
            # 计算yi = w*x+b，b = w0*x0 初始值为1
            wx = 0.0
            for j in range(len(self.w)):
                wx += self.w[j] * x[j]
                
            # 如果wx*y>0,则分类正确，分类正确数+1
            # 如果wx*y<0,则分类错误，需要更新参数
            # wi <- wi+lr*y*xi, b <- b+lr*y
            if wx * y > 0:
                correct_count += 1
                continue
            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])
                
    def predict_point(self,x):
        '''
        预测一个样本
        Parameters:
            x：         单独一个样本点的特征向量
        Return：
            int(wx>0)： x 是否大于 0 的判断
        '''
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx>0)
    

    def predict(self, X):
        '''
        预测所有样本
        Parameters:
            X：        所有样本点的特征矩阵
        Return:
            labels：   返回所有样本的预测
        '''
        labels = []
        for row in X:
            x = row.tolist()
            x.append(1) # 加上 x0
            labels.append(self.predict_point(x))
        return labels

if __name__ == '__main__':

    print('Start read data')
    
    time_1 = time.time()
    raw_data = pd.read_csv(open('./MINST_binary.csv'), header =0)  
   
    # 由于原始数据存在空值，统一替换为0
    raw_data = raw_data.fillna(0.0)
    data = raw_data.values
    features = data[:, 1:]
    labels = data[:, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=0)
    time_2 = time.time()
    
    print('read data cost %f seconds' % (time_2 - time_1))
    print('Start training')

    p = perceptron()
    p.fit(train_features, train_labels)
    time_3 = time.time()

    print('training cost %f seconds' % (time_3 - time_2))
    print('Start predicting')

    test_predict = p.predict(test_features)
    time_4 = time.time()

    print('predicting cost %f seconds' % (time_4 - time_3))
    print('The accruacy score is ', accuracy_score(test_labels, test_predict))







