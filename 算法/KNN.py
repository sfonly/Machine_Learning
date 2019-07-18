# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:33:09 2019

@author: sf_on
"""

import numpy as np
from collections import Counter

# 获取两个向量的距离
def get_distance(vec1, vec2):
    # list转array
    npvec1 = np.array(vec1)
    npvec2 = np.array(vec2)
    distance = ((npvec1 - npvec2)**2).sum()
    return distance

# 获取一个测试向量在训练集上 k 近邻的label    
def get_label(train_feature, test_vec, train_labels, k=1):
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
    result_k = np.array(all_distance_label)
    # 根据距离进行排序(由小到大)
    order_distance = np.argsort(result_k[:,0],axis=0).tolist()
    result_k = result_k[order_distance]
    # 获取前k个点的 距离-类别 数组
    top_k = np.array(result_k[:k,1])
    # 统计预测向量 k个近邻中 最多的label
    label = Counter(top_k).most_common(1)[0][0]
    return str(label)

# 预测训练集的label
def predict(train_feature, test_feature, train_labels, k):
    all_pre_label = []
    for i in range(len(test_feature)):
        test_vec = test_feature[i]
        pre_label = get_label(train_feature, test_vec, train_labels, k)
        all_pre_label.append(pre_label)
    return all_pre_label

# 对于训练集预测的label进行统计
def classify(train_feature, test_feature, train_labels, test_labels, k):
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
