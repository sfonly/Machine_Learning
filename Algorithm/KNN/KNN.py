# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:33:09 2019

KNN Algorithm

@author: sf_on
"""

import numpy as np
from collections import Counter

# 获取两个向量的距离
def get_distance(vec1, vec2):
    '''
    获取两个向量的欧式距离
    Paramters：
        vec1:       向量 1
        vec2:       向量 2
    Return：
        distance：  两个向量的欧式距离
    '''
    npvec1 = np.array(vec1)
    npvec2 = np.array(vec2)
    distance = ((npvec1 - npvec2)**2).sum()
    return distance

def get_label(train_feature, test_vec, train_labels, k=1):
    '''
    获取一个测试向量在训练集上 k 近邻的 label
    Paramters：
        train_feature： 训练数据集特征矩阵
        test_vec:       测试向量
        train_labels:   训练集的类别
        k：             选择多少个近邻
    Return：
        str(prelabel)：    预测这个测试向量的类别
        
    这个过程稍微复杂一点，但是理解起来还是比较容易
    先遍历所有训练集的特征矩阵，计算测试向量和所有训练集特征向量的欧式距离
    根据欧式距离进行排序（将list转为array的原因是，因为list里嵌套了list，不好根据list的值进行排序）
    通过计算一个 array 中前 k 个向量的label进行计算，将其作为预测的 prelabel
    '''
    
    all_distance_label = []
    # 计算测试向量到所有训练向量的距离
    for i in range(len(train_feature)):
        distance_label = []
        train_vec = train_feature[i]
        train_label = train_labels[i]
        vec_distance = get_distance(train_vec, test_vec)
        distance_label.append(vec_distance)
        distance_label.append(train_label)
        all_distance_label.append(distance_label)
    
    # 将距离-类别的list转化为array数组
    # 根据距离进行排序(由小到大)
    result_k = np.array(all_distance_label)
    order_distance = np.argsort(result_k[:,0],axis=0).tolist()
    result_k = result_k[order_distance]
    
    # 获取前k个点的 距离-类别 数组
    # 统计预测向量 k个近邻中 最多的label
    top_k = np.array(result_k[:k,1])
    label = Counter(top_k).most_common(1)[0][0]
    return str(label)

def predict(train_feature, test_feature, train_labels, k):
    '''
    预测测试集（所有测试向量）的label
    Paramters：
        train_feature： 训练数据集特征矩阵
        test_feature:   测试数据集特征矩阵
        train_labels:   训练集的类别
        k：             选择多少个近邻
    Return：
        all_pre_label： 测试集预测的所有类别
    '''
    all_pre_label = []
    for i in range(len(test_feature)):
        test_vec = test_feature[i]
        pre_label = get_label(train_feature, test_vec, train_labels, k)
        all_pre_label.append(pre_label)
    return all_pre_label

def classify(train_feature, test_feature, train_labels, test_labels, k):
    '''
    对于训练集预测的label进行统计
    Paramters：
        train_feature：    训练数据集特征矩阵
        test_feature:      测试数据集特征矩阵
        train_labels:      训练集的类别
        test_labels：      测试集的类别
        k：                选择多少个近邻
    Return：
        true_probability： 测试集的准确率
        all_pre_label：    测试集预测的所有类别
    '''
    error_counter = 0
    pre_labels = predict(train_feature, test_feature, train_labels, k)
    for i in range(len(test_labels)):
        pre_label, test_vec_label = pre_labels[i],test_labels[i]
        if str(test_vec_label) != str(pre_label):
            error_counter += 1
        else:
            continue
    true_probability = 1 - error_counter/len(test_feature)
    return true_probability,pre_labels

if __name__ == '__main__':
    
    # 创建训练集
    train_feature = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    train_labels = ['A','A','B','B']
    test_feature = np.array([[0.9,1.2],[1.1,0.7],[-0.1,0.1],[0.3,0]])
    test_labels = ['A','A','B','B']
    k = 2
    tp,pre_labels = classify(train_feature, test_feature, train_labels, test_labels, k)
    print(tp)
