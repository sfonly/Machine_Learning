# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:55:41 2019

@author: sf_on
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd


def lwlr(testPoint, xArr, yArr, k = 1.0):
    '''
    通过局部加权线性回归计算样本的预测值,只预测一个点
    Paramters：
        testPoint:   预测数据（点）
        xArr:        原始样本特征数组
        yArr:        原始样本真实值数组
    Return:
        testPoint*ws 预测数据的预测值
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    # 高斯核公式计算权重
    for i in range(m):
        diffMat = testPoint - xMat[i,:]
        weights[i,i] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    # 判断xTx是否可逆
    if np.linalg.det(xTx) == 0.0:
        print('can not make xTx^-1')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint*ws


def lwlr_all(testArr, xArr, yArr, k =1.0):
    '''
    循环预测所有样本点
    Paramters：
        testArr：        预测数据集
        xArr：           原始样本特征数组
        yArr：           原始样本真实值数组
    Return：
        yHat：           预测数据集的预测值数组
    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def plotDataSet(xcord,ycord):
    '''
    绘制散点图
    Paramters：
        xcord:          x值
        ycord:          y值
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def plot_lwlr_Regression(xlist,ylist,yhat):
    '''
    绘制曲线图
    Paramters：
        xlist:            x坐标轴
        ylist:           真实 y 值
        yhat：           预测值
    '''
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.plot(xlist,yhat,color='red')
    ax.scatter(xlist,ylist,s=20,c='blue',alpha=0.5)
    plt.show()


if __name__ == '__main__':
    '''
    局部加权回归是非参数学习法
      对每个需要预测的点，都需要进行一次遍历计算
      因此，计算的耗时极长，效率较低
    '''
    
    test = pd.read_csv(open('./test.csv'),
                       header = None, names = ['x0','x1','y'])
    # 根据x1的值进行排序
    test_sort = test.sort_values(by='x1',ascending=True)
    test_x = test_sort[['x0','x1']].values
    test_y = test_sort[['y']].values
    
    xlist = test_x[:,1].tolist()
    ylist = test_y.tolist()
    yhat_k1 = lwlr_all(test_x,test_x,test_y,0.02)
    plot_lwlr_Regression(xlist,ylist,yhat_k1)
