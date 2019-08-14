# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:36:16 2019

支持向量机：SVM

@author: sf_on
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import copy
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def index2isnull(data):
    '''
    查询空值的行号
    Paramters：
        data:       输入的数据集
    Return:
        index:      将存在空值的行号转化为list输出
    '''
    index = []
    for col in data.columns:
        index.extend(data[data[col].isin(['?'])].index.values.tolist())
    return index

def replaceNullvalues(data):
    '''
    用众数去除空值
    Paramters：
        data:       输入的数据集
    Return:
        data_:      用众数替换过后的数据集
    '''
    data_ = copy.deepcopy(data)
    for c in data.columns:    
        data_[c] = data_[c].apply(lambda x: data_[data_[c]!='?'][c].astype(float).mode() if x == '?' else x)
        data_[c] = data_[c].astype(float)  
    return data_

def label_replace(data, label, new_label):
    '''
    类标号处理
    将[0,1,2,3]的类标号，大于 1 的列表转换为 1
    Paramters：
        data:       输入的数据集
        label：     类标号
    Return:
        data:       替换过后的类标号
    '''
    set(data.loc[:, label].values)
    data.loc[:, new_label] = data.loc[:,label].apply(lambda x: 1 if x >=1 else 0)
    return data

def scatter_features_matrics(data, colomns,label):
    '''
    将连续特征进行矩阵显示
    连续特征： 
        'age','blood pressure','serum_cholestoral','max_heart_rate','ST_depression'
    离散特征：
        'sex','chest_pain','fasting_blood_sugar','electrocardiographic',
        'induced_angina', 'slope', 'vessels', 'thal'
    类标号：
        'diagnosis', 'diag_int'
    Paramters：
        data:           输入的数据集
        colomns：       连续特征
        label：         列表号  
    '''
    features = data[colomns]
    features[label] = features[label].replace([0,1],['blue','g']) 
    pd.plotting.scatter_matrix(features.iloc[:,:5], alpha=0.7, c=features.iloc[:,5],figsize=(12,12), diagonal='hist')
    plt.show()

def scatter_xy(x,y,data):
    '''
    数据可视化分析
    Paramters：
        x:          横坐标的列
        y:          纵坐标的列
        data:       数据集
    '''
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Relationship between' + x +' and ' + y)
    plt.scatter(data[x], data[y])
    plt.show()

def show_corr(data):
    '''
    相关系数图热力图
    Paramters：
        data:       数据集
    '''
    cor = data.corr() 
    plt.figure(figsize=(12,10))
    sns.heatmap(cor)  
    plt.show()

def data_standarlize(data, columns):
    '''
    数据标准化
    Paramters：
        data:            输入数据集
        columns:         需要标准化的列
    Return:
        standare_data：  标准化后的数据集  
    '''
    standare_data = preprocessing.scale(data_new.loc[:,columns])
    standare_data = pd.DataFrame(standare_data, columns=columns)
    return standare_data

def function_classification(x,y):
    '''
    利用 SVC 方法对数据进行分类并测试
    Paramters：
        x:           特征矩阵
        y:           类标号  
    Return:
        expected：   期望的类标号
        predicted：  预测的类标号
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.33,random_state=6)
    model = SVC(kernel = 'linear')
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    f1 = f1_score(expected, predicted)
    tn,fp,fn,tp = confusion_matrix(expected, predicted, sample_weight=None).ravel()
    
    print('accuracy score: ', model.score(x_test,y_test))
    print('f1 score: ', f1)
    print('tp = ' + str(tp) + ' , fp = ' + str(fp))
    print('fn = ' + str(fn) + ' , tn = ' + str(tn))
    print(metrics.classification_report(expected, predicted)) 
    return expected,predicted 

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    绘制混淆函数的可视化方法
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def function_plot(expected,predicted):
    '''
    绘制混淆函数的可视化方法
    Paramters：
        expected:       期望的类标号
        predicted:      预测的类标号
    '''
    cnf_matrix = confusion_matrix(expected, predicted)
    plot_confusion_matrix(cnf_matrix, 
                          classes=['Heart disease', 'No heart disease'],
                          title='Confusion matrix')
    plt.show()   

def output_csv(x,y):
    '''
    导出训练文件和测试文件
    Paramters：
        x:       特征矩阵
        y:       类标号
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.33,random_state=6)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    Traindata = pd.merge(x_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/heartdisease_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(x_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/heartdisease_test.csv',index = False,encoding = 'utf-8')


if __name__ == '__main__':
    
    header_row = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',
                   'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal','diagnosis']
    data = pd.read_csv(open('./heart_disease.csv'),header = None, names = header_row)
    index_ = index2isnull(data)
    
    print('data.shape: ',data.shape)
    print('data.describe():',data.describe())
    print('index_:',index_)
    print('index_.length:',len(index_))
    
    # 替换数据集中的空值
    data_new = replaceNullvalues(data)
    # 处理数据集的类标号
    data_new = label_replace(data_new, 'diagnosis', 'diag_int')
    # 相关性分析
    corr = data_new.corr()
    show_corr(data_new)
    
    # 将连续特征进行矩阵展示
    continus_features = ['age','blood pressure','serum_cholestoral','max_heart_rate','ST_depression','diag_int']
    scatter_features_matrics(data_new, continus_features, 'diag_int')
    
    test = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',
                   'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal']
    
    standarlize = data_standarlize(data_new, test)
    
    expected,predicted = function_classification(standarlize,data_new['diag_int'])
    function_plot(expected,predicted)
    
    # output_csv(x_standardized,data_new['diag_int'])


