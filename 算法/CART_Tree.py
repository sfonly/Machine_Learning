# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:44:07 2019

@author: sf_on
"""

import pandas as pd
import operator

# 读取csv数据
def loadDataSet(url, name, columns):
    path = url
    dataSet = pd.read_csv(path + name, header = None, names = columns)
    return dataSet

# 根据label类标号，统计dataSet中不同类标号的个数，返回字典类型counts
# 如果label类标号必须是倒数第一列，统计不同类标号里面有多少行输出格式，如二分类的情况： {[0,1],[1,0],[0,0]...}
def classCount(dataSet):
    counts = {}
    labels = dataSet.iloc[:, -1]
    for one in labels:
        if one not in counts.keys:
            counts[one] = 0
        counts[one] += 1
    return counts

# gini系数计算公式: gini(事件P) = 1 - p1^2 - p2^2
def calcGini(dataSet):
    gini = 1.00
    counts = classCount(dataSet)
    for one in counts.keys():
        prob = float(counts[one])/len(dataSet)
        gini -= (prob*prob)
    return gini

# 数据集排序，根据类标号的多少进行排序，sortedCounts[0][0]为最多的一个类别
def majorityClass(dataSet):
    counts = classCount(dataSet)
    sortedCounts = sorted(counts.items(), key = operator.itemgetter(1), reverse = True)
    return sortedCounts[0][0]

# 分割数据,根据节点的筛选条件，将原来的大表切分为小的表
# 第i列属性，value为比较值，direction为方向
def splitDataSet(dataSet, i, value, direction):
    # 判断数据类型，如果数据类型是连续值，则通过大小进行切割数据
    if type(dataSet.iloc[0, i]).__name__=='float64':
        # 方向为0，代表右子节点
        if direction == 0:
            # 将第i列中大于value的数据收集起来
            subDataSet = dataSet[dataSet.iloc[:,i] > value]
        # 方向为1，代表左子节点
        if direction == 1:
            # 将第i列中小于value的数据收集起来
            subDataSet = dataSet[dataSet.iloc[:,i] <= value]
    # 判断数据类型，如果数据类型是标称值，则通过类别进行切割数据
    if type(dataSet.iloc[0, i]).__name__=='str':
        if direction == 0:
            # 将第i列中和value相同的数据收集起来
            subDataSet = dataSet[dataSet.iloc[:,i] == value]
        if direction == 1:
            # 将第i列中和value不同的数据收集起来
            subDataSet = dataSet[dataSet.iloc[:,i] != value]        
    # 第i列只用一次，因此drop掉，实际上在CART的优化算法中，满足分割要求的第 i 列可以改完多次使用
    reduceDataSet = subDataSet.drop(subDataSet.columns[i],axis=1)
    return reduceDataSet

# 计算第i个特征，找出其中最优的区分点
def chooseFeature(dataSet,i):
    # 找出第i列的不同的各类属性值
    valueList = set(dataSet.iloc[:,i])
    bestSplitGini = 10.0
    for value in valueList:
        newGiniIndex = 0.0
        # 放入右侧的数据集
        greaterDataSet = splitDataSet(dataSet,i,value,0)
        # 放入右侧的数据集
        smallerDataSet = splitDataSet(dataSet,i,value,1)            
        # 计算右子节点发生的可能性
        prob0 = float(len(greaterDataSet))/len(dataSet)
        # 计算左子节点发生的可能性
        prob1 = float(len(smallerDataSet))/len(dataSet)
        # 计算当前节点第i个属性的gini系数
        newGiniIndex += prob0 * calcGini(greaterDataSet) + prob1 * calcGini(smallerDataSet)
        # 找出最优切割点，gini系数越小，数据越纯净，即是我们想要的切割点               
        if newGiniIndex < bestSplitGini:
            bestSplitGini = newGiniIndex
            bestSplitValue = value
    print('tempBestFeat:'+str(dataSet.columns[i])+' ,GiniIndex:'+str(newGiniIndex))
    return bestSplitGini,bestSplitValue


# 寻找最优的特征来分割数据
def chooseBestFeat(dataSet):
    # 用于记录最佳属性值是第几列
    bestFeat = 0
    # 用于装入最佳的属性的属性值
    splitDic = {}
    # 初始化一个最优的gini系数
    bestGiniIndex = 100.00
    GiniIndex = 100.0
    # 遍历除类标号之外的所有列
    for i in range(len(dataSet.iloc[0,:]-1)):
        GiniIndex,splitDic[dataSet.columns[i]] = chooseFeature(dataSet,i)
        # 找出最佳的分割属性
        if GiniIndex < bestGiniIndex:
            bestGiniIndex = GiniIndex
            bestFeat = i
        # 找出最佳分割属性的分割点或者是属性值
        bestFeatValue = splitDic[dataSet.columns[bestFeat]]
    return bestFeat, bestFeatValue

# 生成树模型，递归的形式生成树模型
def createTree(dataSet):
    # 最后生成的树模型，是字典形式
    myTree = {bestFeatLabel:{}}
    # 计算当前数据集的gini系数，如果要进行剪枝，可以使用
    gini = calcGini(dataSet)
    # 如果数据集的只有一列，也就是叶子节点，返回类别数量
    if len(dataSet.columns) == 1:
        return majorityClass(dataSet)
    # 如果是已经分得纯洁了，是叶子节点，停止    
    if len(set(dataSet.iloc[:,-1])) == 1:
        return dataSet.iloc[0,-1]
    bestFeat,bestFeatValue = chooseBestFeat(dataSet)
    bestFeatLabel = dataSet.columns[bestFeat]
    print('bestFeat:'+str(bestFeatLabel))
    print('bestFeatValue:'+str(bestFeatValue))
    # 如果数据集中的列是连续值
    if type(dataSet.iloc[0, bestFeat]).__name__=='float64':
        greaterDataSet = splitDataSet(dataSet,bestFeat,bestFeatValue,0)
        smallerDataSet = splitDataSet(dataSet,bestFeat,bestFeatValue,1)
        myTree[bestFeatLabel]['>'+str(bestFeatValue)] = createTree(greaterDataSet)
        myTree[bestFeatLabel]['<='+str(bestFeatValue)] = createTree(smallerDataSet)
    # 如果数据集中的列是标称属性
    if type(dataSet.iloc[0, bestFeat]).__name__=='str':
        greaterDataSet = splitDataSet(dataSet,bestFeat,bestFeatValue,0)
        smallerDataSet = splitDataSet(dataSet,bestFeat,bestFeatValue,1)
        myTree[bestFeatLabel]['=='+str(bestFeatValue)] = createTree(greaterDataSet)
        myTree[bestFeatLabel]['!='+str(bestFeatValue)] = createTree(smallerDataSet)
    return myTree

# 主函数，程序入口
if __name__ == '__main__':
    dataSet = loadDataSet(url, filename, columns)
    dataSet._convert(float)
    print(createTree(dataSet))
