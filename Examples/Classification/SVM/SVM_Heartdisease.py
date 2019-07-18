# -*- coding: utf-8 -*-
# Filename: SVM_Heartdisease.py
"""
Created on Mon Jun 17 10:36:16 2019

@author: sf_on
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

header_row = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',
               'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal','diagnosis']
data = pd.read_csv(open('./heart_disease.csv'),header = None, names = header_row)
data.shape

# 查看数据集中的空值
index = []
for col in data.columns:
    index.extend(data[data[col].isin(['?'])].index.values.tolist())
print(index)

# 用众数去除空值,apply函数、lambda函数的用法，核心是lambda函数            
for c in data.columns:    
    data[c] = data[c].apply(lambda x: data[data[c]!='?'][c].astype(float).mode() if x == "?" else x)
    data[c] = data[c].astype(float)  
set(data.loc[:, "diagnosis"].values)
data.loc[:, "diag_int"] = data.loc[:,"diagnosis"].apply(lambda x: 1 if x >=1 else 0)


corr = data.corr()

# 数据可视化分析
def scatter_xy(x,y,data):
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Relationship between" + x +" and " + y)
    plt.scatter(data[x], data[y])
    plt.show()
scatter_xy('diagnosis','diag_int',data)
feature = data[['age','sex','chest_pain','blood pressure','serum_cholestoral',
                          'fasting_blood_sugar','electrocardiographic','max_heart_rate',
                          'induced_angina','ST_depression','slope','vessels','thal','diag_int']]
pd.plotting.scatter_matrix(feature, alpha=0.7, figsize=(24,24), diagonal='hist')

# 数据标准化，并且将标准化后的数据转化为dataframe格式
from sklearn import preprocessing
x_standardized = preprocessing.scale(data.iloc[:,0:13])
x_standardized = pd.DataFrame(x_standardized)
x_standardized.columns = ['age','sex','chest_pain','blood pressure','serum_cholestoral',
                          'fasting_blood_sugar','electrocardiographic','max_heart_rate',
                          'induced_angina','ST_depression','slope','vessels','thal']

from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

# 划分训练集和测试集的方法
def selection(x,y):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.33,random_state=6)
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = selection(x_standardized,data.iloc[:,14])

# 绘制混淆函数的可视化方法
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 绘制混淆函数的可视化方法
def function_plot(expected,predicted):
    cnf_matrix = confusion_matrix(expected, predicted)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["Heart disease", "No heart disease"],
                      title='Confusion matrix')
    plt.show()   

# 利用SVC方法对数据进行分类并测试
def function_classification(x,y,function,typemode):
    # 切分数据
    x_train, x_test, y_train, y_test = selection(x,y)
    model = function(kernel = typemode)
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    f1 = f1_score(expected, predicted)
    tn,fp,fn,tp = confusion_matrix(expected, predicted, sample_weight=None).ravel()
    print(model.score(x_test,y_test))
    print(f1)
    print(metrics.classification_report(expected, predicted)) 
    print(confusion_matrix(expected, predicted, sample_weight=None))
    print('tp = ' + str(tp) + ' , fp = ' + str(fp))
    print('fn = ' + str(fn) + ' , tn = ' + str(tn))
    return expected,predicted 

expected,predicted = function_classification(x_standardized,data['diag_int'],SVC,'linear')
function_plot(expected,predicted)

# 导出训练文件和测试文件
def output_csv(x,y):
    x_train, x_test, y_train, y_test = selection(x,y)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    Traindata = pd.merge(x_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/heartdisease_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(x_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/heartdisease_test.csv',index = False,encoding = 'utf-8')
output_csv(x_standardized,data['diag_int'])


