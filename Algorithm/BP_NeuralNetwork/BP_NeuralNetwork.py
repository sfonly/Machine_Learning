# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:48:04 2019

BP神经网络（误差反向传播神经网络）
这里只实现了三层网络

@author: sf_on
"""

import numpy as np
import math
import random


def random_number(a,b):
    '''
    产生一个在（a,b）之间的随机数
    Paramters：
        a,b:        取值范围      
    Return：        a到 b之间的随机数
    '''
    return (b-a)*random.random()+a

def makematrix(m, n, fill=0.0):
    '''
    产生一个 m * n 维的取值全是 0 的矩阵
    Paramters：
        m,n:        设置矩阵维度 m * n
        fill:       设置填充值
    Return:
        matrix:     返回矩阵
    '''
    matrix = []
    for i in range(m):
        matrix.append([fill]*n)
    return matrix

def sigmoid(x):
    '''
    sigmoid 工具函数
    Paramters：
        x:          输入
    Return:         sigmoid函数返回值
    '''
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    '''
    sigmoid 的导数
    Paramters：
        x:          输入
    Return:         sigmoid函数的导数的返回值
    '''
    return x * (1 - x)

def Normalization(x):
    '''
    最大最小标准化，针对于某一列
    Paramters：
        x:          输入特征矩阵
    Return:         返回最大最小标准化后的输入矩阵
    '''
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def list_Normalization(features):
    '''
    list对象的标准化
    Paramters：
        features：   输入特征，list对象
    Return：
        outer：      输出
    '''
    features = np.array(features)
    for col in range(len(features[0])):
        features[:,col] = Normalization(features[:,col])
    outer = features.tolist()
    return outer

def data_preprocessing(data):
    '''
    数据预处理
    '''
    features = []
    for row in data:
        features.append(row[0])
    features = list_Normalization(features)
    for i in range(len(features)):
        data[i][0] = features[i]


class BP:
    
    
    def __init__(self, ni, nh, no):
        '''
        初始化神经网络
        Paramters：
            ni:         输入层的节点个数
            nh:         隐藏层的节点个数
            no:         输出层的节点个数
        '''
        
        # 初始化三层神经网络的神经元个数
        self.input_n = ni + 1 # 输入层的节点个数, 加上x0=1的输入
        self.hidden_n = nh    # 隐藏层的节点个数
        self.output_n = no   # 输出层的节点个数
        
        # 初始化三层的输入矩阵
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n        
        self.output_cells = [1.0] * self.output_n
        
        # 初始化神经网络权重 W[i][j]的矩阵        
        self.input_weights = makematrix(self.input_n, self.hidden_n)
        self.output_weights = makematrix(self.hidden_n, self.output_n)
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = random_number(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = random_number(-2.0, 2.0)
                
        # 初始化误差矩阵
        self.input_correction = makematrix(self.input_n, self.hidden_n)
        self.output_correction = makematrix(self.hidden_n, self.output_n)  


    def predict(self, inputs):
        '''
        向前传播算法
        Paramters：
            inputs:                   神经网络的输入
        Return:
            self.output_cells[:]：    神经网络输出层输出
        '''
        
        # 激活输入层，将输入 inputs 传入输入的cell中
        for i in range(self.input_n -1):
            self.input_cells[i] = inputs[i]
            
        # 激活隐含层            
        for j in range(self.hidden_n):
            sum = 0.0
            # 计算隐含层第j个节点的net输入值
            for i in range(self.input_n):
                sum = sum + self.input_cells[i] * self.input_weights[i][j]      
            self.hidden_cells[j] = sigmoid(sum)

        # 激活输出层       
        for k in range(self.output_n):
            total = 0.0 
            # 计算输出层第k个节点的net输入值            
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        
        # 输出层输出         
        return self.output_cells[:]

             
    def back_propagate(self, case, label, learn, correct):
        '''
        误差向后传播算法
        Paramters：
            case:           神经网络的输入
            label:          类标号
            learn:          学习速率
            correct:        误差系数，用于控制误差反向传播的修正速率
        Return:
            error:          误差
        '''
        # 执行向前传播
        self.predict(case)
        
        # 计算每一个输出神经元的误差以及误差的敏感度 out_delta        
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        
        # 计算每一个隐含层神经元的误差以及误差的敏感度 hidden_delta         
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            # 计算隐含层神经元的误差
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            # 计算隐含层神经元的误差敏感度
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error

        # 更新隐含层和输出层所有连线的权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                # change为每次迭代的梯度，这里用于改进BP算法
                # 引入冲量，使得权值调整具有一定惯性
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # 更新输入层和隐含层所有连线的权重        
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                # change为每次迭代的梯度，这里用于改进BP算法
                # 引入冲量，使得权值调整具有一定惯性          
                change = hidden_deltas[h] * self.input_cells[i]           
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        
        # 计算每次迭代的总误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 *(label[o] - self.output_cells[o])**2
            
        return error


    def train(self, pattern, itera=20000, learn = 0.1, correct = 0.1):
        '''
        训练模型，并打印每次迭代的模型总误差
        Paramters：
            pattern：        预测集特征矩阵
            itera：          迭代次数
            learn：          学习速率
            correct：        误差系数，用于控制误差反向传播的修正速率
        '''
        for i in range(itera):
            error = 0.0
            for j in pattern:
                inputs = j[0]
                targets = j[1]
                error = error + self.back_propagate(inputs, targets,learn, correct)
            if i % 100 == 0:
                print('迭代次数 %s ,当前总误差为： %.8f' %(i,error))


    def test(self, patterns):
        '''
        测试，输出神经网络预测结果
        Paramters：
            patterns:       预测集特征矩阵
        '''
        count = 0
        for i in patterns:
            count += 1
            label = self.predict(i[0])
            feature = i[0]
            print('第 %s 个预测结果:' % count ,np.around(feature,decimals =5) , '->', np.around(label,decimals =5))


    def weights(self):
        '''
        输出神经网络每一层，所有节点的权重
        '''
        print('输入层到隐含层的权重：')
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                 print('第 %s 个输入层节点，到第 %s 隐含层节点的权重为： %.8f' %(i,h+1,self.input_weights[i][h]))
           
        print('隐含层到输出层的权重：')
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                print('第 %s 个隐含层节点，到第 %s 输出层节点的权重为： %.8f' %(h+1,o+1,self.output_weights[h][o]))
                
    

if __name__ == '__main__':

    data = [
            [[3.2,  96, 3.45, 2.15, 1.4, 2.8, 1.10, .50],[1]],
            [[3.2, 103, 3.75, 2.20, 1.2, 3.4, 1.09, .70],[1]],
            [[3.0,  90, 3.50, 2.20, 1.4, 3.5, 1.14, .50],[1]],
            [[3.2, 103, 3.65, 2.20, 1.5, 2.8, 1.08, .80],[1]],
            [[3.2, 101, 3.50, 2.00, 0.8, 1.5, 1.13, .50],[0]],
            [[3.4, 100, 3.40, 2.15, 1.3, 3.2, 1.15, .60],[1]],
            [[3.2,  96, 3.55, 2.10, 1.3, 3.5, 1.18, .65],[0]],
            [[3.0,  90, 3.50, 2.10, 1.0, 1.8, 1.13, .40],[1]],
            [[3.2,  96, 3.55, 2.10, 1.3, 3.5, 1.18, .65],[0]],
            [[3.2,  92, 3.50, 2.10, 1.4, 2.5, 1.10, .50],[1]],
            [[3.2,  95, 3.40, 2.15,1.15, 2.8, 1.19, .50],[1]],
            [[3.9,  90, 3.10, 2.00, 0.8, 2.2, 1.30, .50],[0]],
            [[3.1,  95, 3.60, 2.10, 0.9, 2.7, 1.11, .70],[0]],
            [[3.2,  97, 3.45, 2.15, 1.3, 4.6, 1.09, .70],[1]]
            ]

    # 数据预处理，将data的特征正规化
    data_preprocessing(data)
        
    #创建神经网络，8个输入节点，10个隐藏层节点，1个输出层节点
    n = BP(8, 10, 1)
    # 训练并测试神经网络，查看权重值
    n.train(data)
    n.test(data)
    n.weights()
