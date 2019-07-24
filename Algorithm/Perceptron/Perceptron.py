# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:39:17 2019

@author: sf_on
"""

import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class perceptron(object):
    def __init__(self):
        # 学习率
        self.learning_step = 0.001
        # 分类正确上界，当分类正确的次数超过上界时，认为已训练好，退出训练
        self.max_iteration = 10000 
        
    def train(self, features, labels):
        # 加上w0
        self.w = [0.0] * (len(features[0])+1)
        # 分类正确数，用于控制循环次数
        correct_count = 0
        while correct_count < self.max_iteration:
            # 采用随机梯度下降，每次只更新一个样本点
            index = random.randint(0, len(labels) -1)
            x = list(features[index])
            # 加上x0
            x.append(1.0)
            # 和SVM类似，将问题转化为 y*(wx+b)=1 的问题，因此 y=1或者-1
            y = 2 * labels[index] -1
            wx = 0.0
            # 计算yi = w*x+b，b = w0*x0 初始值为1
            for j in range(len(self.w)):
                wx += self.w[j] * x[j]
                
            # 如果wx*y>0,则分类正确，分类正确数+1
            if wx * y > 0:
                correct_count += 1
                continue
            # 如果wx*y<0,则分类错误，需要更新参数
            # w <- w + lr*y*xi, b <- b + lr*y
            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])
                
    # 预测一个样本
    def predict_(self,x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx >0)
    
    # 预测n个样本    
    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

if __name__ == '__main__':

    print('Start read data')
    time_1 = time.time()
    
    # 读取csv数据，并将第一行视为表头，返回DataFrame类型
    raw_data = pd.read_csv(open('C://Users/sf_on/Desktop/数据挖掘应用分析实验手册/感知机/MINST_binary.csv'), header =0)  
    # 由于原始数据存在空值，统一替换为0
    raw_data = raw_data.fillna(0.0)
    data = raw_data.values

    features = data[:, 1:]
    labels = data[:, 0]

    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training')
    p = perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print('The accruacy score is ', score)
