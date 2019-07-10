# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:48:04 2019

@author: sf_on
"""

import numpy as np
import math
import random

def random_number(a,b):
    return (b-a)*random.random()+a

def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill]*n)
    return a

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class BP:
    # 初始化神经网络
    def __init__(self, ni, nh, no):
        # 初始化三层神经网络的神经元个数
        self.input_n = ni + 1 # 输入层的节点个数,加一是加上X0=1的输入
        self.hidden_n = nh    # 隐藏层的节点个数
        self.output_n = no   # 输出层的节点个数

        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n        
        self.output_cells = [1.0] * self.output_n
        # 初始化神经网络权重 W[i][j]的矩阵        
        self.input_weights = makematrix(self.input_n, self.hidden_n)
        self.output_weights = makematrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = random_number(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = random_number(-2.0, 2.0)
        # 初始化偏置
        self.input_correction = makematrix(self.input_n, self.hidden_n)
        self.output_correction = makematrix(self.hidden_n, self.output_n)  

    # 向前传播，计算每层的输出    
    def predict(self, inputs):
        # 激活输入层，将 x 传入输入的cell中
        for i in range(self.input_n -1):
            self.input_cells[i] = inputs[i]
        # 激活隐含层            
        for j in range(self.hidden_n):
            sum = 0.0
            # 计算隐含层第j个节点的net输入值
            for i in range(self.input_n):
                sum = sum + self.input_cells[i] * self.input_weights[i][j]
            # 计算隐含层第j个节点的out输出值          
            self.hidden_cells[j] = sigmoid(sum)

        # 激活输出层       
        for k in range(self.output_n):
            total = 0.0 
            # 计算输出层第k个节点的net输入值            
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            # 计算输出层第k个节点的out输出值 
            self.output_cells[k] = sigmoid(total)
        # 输出层输出         
        return self.output_cells[:]

    # 误差向后传播               
    def back_propagate(self, case, label, learn, correct):
        # 向前传播
        self.predict(case)
        # 获取每一个输出神经元的误差以及误差的敏感度 out_delta        
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            # 输出层神经元敏感度计算公式： delta = E * oj * (1-oj)
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        
        # 获取每一个隐含层神经元的误差以及误差的敏感度 hidden_delta         
        hidden_deltas = [0.0] * self.hidden_n
        # 遍历每一个隐含层神经元
        for h in range(self.hidden_n):
            error = 0.0
            # 隐含层神经元的误差，等于所有输出层的误差反向传播到该神经元的误差
            for o in range(self.output_n):
                # 误差计算公式：e = each_sum(out_delta * w[h][o])
                error += output_deltas[o] * self.output_weights[h][o]
            # 隐含层神经元敏感度计算公式： hidden_delta = E * oj * (1-oj)
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error

        # 遍历隐含层和输出层所有连线的权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                # change为每次迭代的梯度
                change = output_deltas[o] * self.hidden_cells[h]
                # 改进BP算法，加入冲量，使得权值调整具有一定惯性
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # 遍历输入层和隐含层所有连线的权重        
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                # change为每次迭代的梯度                
                change = hidden_deltas[h] * self.input_cells[i]
                # 改进BP算法，加入冲量，使得权值调整具有一定惯性                
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 设置BP每次迭代的总误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 *(label[o] - self.output_cells[o])**2
        return error
                
    def train(self, pattern, itera=20000, learn = 0.1, correct = 0.1):
        for i in range(itera):
            error = 0.0
            for j in pattern:
                inputs = j[0]
                targets = j[1]
                error = error + self.back_propagate(inputs, targets,learn, correct)
            if i % 100 == 0:
                print('误差 %-.5f' % error)
    def test(self, patterns):
        for i in patterns:
            print(i[0], '->', self.predict(i[0]))
    #权重
    def weights(self):
        print("输入层权重")
        for i in range(self.input_n):
            print(self.input_weights[i])
        print("输出层权重")
        for i in range(self.hidden_n):
            print(self.output_weights[i])        
    
def demo():
    patt = [
            [[1,2,5],[0]],
            [[1,3,4],[1]],
            [[1,6,2],[1]],
            [[1,5,1],[0]],
            [[1,8,4],[1]]
            ]
    #创建神经网络，3个输入节点，3个隐藏层节点，1个输出层节点
    n = BP(3, 5, 1)
    #训练神经网络
    n.train(patt)
    #测试神经网络
    n.test(patt)
    #查阅权重值
    n.weights()

test = [
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

import copy
def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

X = []
data = copy.deepcopy(test)
for i in range(len(data)):
    X.append(data[i][0])
x = np.array(X)
m =  x[:,0]
for col in range(len(x[0])):
    x[:,col] = Normalization(x[:,col])
x = x.tolist()
for i in range(len(data)):
    data[i][0] = x[i]

#创建神经网络，8个输入节点，10个隐藏层节点，1个输出层节点
n = BP(8, 10, 1)
#训练神经网络
n.train(data)
#测试神经网络
n.test(data)
#查阅权重值
n.weights()


