# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:24:41 2019

@author: sf_on
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

x,y = make_blobs(n_samples=1000, n_features=2 , centers = [[-1,1],[0,0],[1,1],[2,2]],
                 cluster_std = [0.4,0.3,0.4,0.3], random_state = 9)

plt.scatter(x[:,0], x[:,1], marker='o', c=y)
plt.show()

from sklearn.cluster import Birch
predicted = Birch(n_clusters = 4).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=predicted)
plt.show()

from sklearn import metrics
print(metrics.calinski_harabaz_score(x,predicted))

param_grid = {'threshold':[0.3,0.2,0.1], 'branching_factor':[50,40,30,20,10]}

param_data = pd.DataFrame(columns = ['threshold', 'branching_factor'])
param_data.threshold = [0.1, 0.2, 0.2, 0.2, 0.3]
param_data.branching_factor = [10, 10,  30,  40, 10]

for threshold,branching_factor in zip(param_data.threshold,param_data.branching_factor):
        clf = Birch(n_clusters = 4, threshold = threshold, branching_factor = branching_factor )
        clf.fit(x)
        predicted = clf.predict(x)
        plt.scatter(x[:,0],x[:,1],c = predicted)
        plt.show()
        print('threshold:', threshold, 'branching_factor:', branching_factor, metrics.calinski_harabaz_score(x,predicted))

print(metrics.calinski_harabaz_score(x,y))

clf = Birch(n_clusters = 4, threshold = 0.15, branching_factor = 20 )
clf.fit(x)
predicted = clf.predict(x)
print(metrics.calinski_harabaz_score(x,predicted))
plt.scatter(x[:,0],x[:,1],c = predicted)
plt.show()
