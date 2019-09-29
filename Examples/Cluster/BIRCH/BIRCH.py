# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:24:41 2019

Adaboost

@author: sf_on
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from sklearn import metrics

def plot_scatter(x, y):
    '''
    打印散点图
    Parameters：
        x:      x轴坐标
        y:      y轴坐标
    '''
    plt.scatter(x[:,0], x[:,1], marker='o', c=y)
    plt.show()


def Grid_Birch(param_grid, features):
    '''
    BIRCH 基于密度聚类
    Parameters：
        param_grid：     参数网格
        x：              特征    
    '''
    for threshold, branching_factor in zip(param_grid['threshold'], param_grid['branching_factor']):
        clf = Birch(n_clusters = 4, threshold = threshold, branching_factor = branching_factor)
        clf.fit(features)
        predicted = clf.predict(features)
        
        plot_scatter(features, predicted)
        print('threshold:', threshold, 'branching_factor:', branching_factor)
        print('metrics.calinski_harabaz_score:', metrics.calinski_harabaz_score(features,predicted))

if __name__ == '__main__':

    x,y = make_blobs(n_samples = 1000, centers = [[-1,1],[0,0],[1,1],[2,0]],
                     cluster_std = [0.4,0.3,0.4,0.3], random_state = 9)
    plot_scatter(x, y)
    print('This is the true blobs scatter')

    
    param_grid = {'threshold':[0.1, 0.2, 0.2, 0.2, 0.3], 'branching_factor':[10, 10,  30,  40, 10]}
    Grid_Birch(param_grid, x)
    
    clf = Birch(n_clusters = 4, threshold = 0.15, branching_factor = 20 )
    clf.fit(x)
    predicted = clf.predict(x)

    plot_scatter(x, predicted)
    print('threshold:', 0.15, 'branching_factor:', 20)
    print('metrics.calinski_harabaz_score(x,predicted):',metrics.calinski_harabaz_score(x,predicted))

    plot_scatter(x, y)
    print('threshold:', 0.15, 'branching_factor:', 20)
    print('metrics.calinski_harabaz_score(x,y):',metrics.calinski_harabaz_score(x,y))




