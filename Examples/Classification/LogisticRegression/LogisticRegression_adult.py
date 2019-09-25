# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:55:22 2019

Logistic_Regression

@author: sf_on
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN


def deletNull(data, colmns):
    '''
    删除 data 中存在空值的列
    Parameters:
        data：           数据集
        colmns：         要去除空值的列的list
    '''
    i_ = []
    for col in columns:
        i_.extend(data[data[col].isnull()].index.tolist())
        
    i_ = list(set(i_))
    data.drop(i_,axis=0,inplace=True)


def kdeplot_rich(data, col):
    '''
    曲线图
    Parameters:
        data：           数据集
        col：            列
    '''
    plt.figure(figsize=(9,4))
    g = sns.kdeplot(data[col][(data['rich']==0)],color='Red',shade=True)
    g = sns.kdeplot(data[col][(data['rich']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel(col)
    g.set_ylabel('Frequency')
    g = g.legend(['poor','rich'])
    plt.show()


def barplotrich(data, col):
    '''
    柱状图
    Parameters:
        data：           数据集
        col：            列
    '''
    order = orderlist(data,col) 
    
    if col == 'education':
        plt.figure(figsize=(15,4))
    elif col == 'occupation':
        plt.figure(figsize=(22,4))
    elif col == 'marital_status':
        plt.figure(figsize=(11,4))
    elif col == 'native_country':
        plt.figure(figsize=(50,4))
    else:
        plt.figure(figsize=(8,4))
        
    g = sns.barplot(data = data,x = col,y = 'rich',order=order)
    g.set_ylabel('Survival Probability')
    plt.show()


def orderlist(data, col):
    '''
    对 col 进行排序，并输出排序后的 index
    Parameters:
        data：           数据集
        col：            列
    Return：
        order：          排序过后的list
    '''
    sort_list = []
    datasort_ = data[[col,'rich']]
    grouped = datasort_.groupby(col)
    
    for key, group in grouped:
        temp = []
        num_rich = group[group['rich'] == 1].shape[0]
        num_total = group.shape[0]
        probability = float(num_rich/num_total)
        temp.append(key)
        temp.append(probability)
        sort_list.append(temp)
        
    sort_list.sort(key=(lambda x:x[1]))
    sort_array = np.array(sort_list)
    order = sort_array[:,0].tolist()
    print(sort_array)
    
    return order


def show_corr(cor):
    '''
    展示相关性系数矩阵热力图
    Parameters:
        cor:        皮尔逊相关系数矩阵
    '''
    plt.figure(figsize=(20,16))
    sns.heatmap(cor)  
    plt.show()


def feature_dummies(data, col, dic):
    '''
    处理离散特征，将其映射到对应组别，并进行哑变量处理
    Parameters:
        data：       数据集
        col:        皮尔逊相关系数矩阵
        dic:        存储特征的映射字典
    Return：
        data：       哑变量处理后的数据集
    '''
    data[col] = data[col].map(dic)
    data = pd.get_dummies(data, columns=[col],prefix= col)
    return data


def logistic_classification(x,y):
    '''
    训练模型，并进行测试
    Parameters：
        x,y:        特征与类标号
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y,test_size=0.3,random_state=36)
    
    model = LogisticRegression()
    mf = model.fit(x_train, y_train)
    expected = y_test
    predicted = mf.predict(x_test)
    f1 = f1_score(expected, predicted)
    
    print('accuracy_score：', mf.score(x_test,y_test))
    print('f1_score：', f1)
    print(metrics.classification_report(expected, predicted)) 
    
    tn,fp,fn,tp = confusion_matrix(expected, predicted, sample_weight=None).ravel()
    y_score = mf.decision_function(x_test)
    fpr,tpr,threshold = metrics.roc_curve(y_test,y_score)
    roc_auc = metrics.auc(fpr,tpr) # 计算auc的值
    show_Roc(fpr,tpr,roc_auc)


def show_Roc(fpr,tpr,roc_auc):
    '''
    绘制 ROC 曲线
    Parameters：
        fpr：            假阳性率，False Positive rate
        tpr：            真阳性率，true positive rate
        roc_auc：        auc 的值，面积
    '''
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    
    columns=['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
             'relationship','race','sex','capital_gain','capital_loss','hours_per_week' ,'native_country','rich']
    data = pd.read_csv(open('./adult_data.csv'),header = None, names = columns)
    
    # 查看数据集中是否存在空值
    print(data.info())
    print(data.isnull().sum())
    
    # 去除空值
    null_cols = ['workclass','occupation','native_country']
    deletNull(data, null_cols)
    
    # 查看数据特征
    print('poor: ' + str(data[data.rich ==0].index.size))
    print('rich: ' + str(data[data.rich ==1].index.size))
    
    # 特征工程
    # ----------------------------------------------------
    # age
    kdeplot_rich(data, 'age')
    
    # education_num
    barplotrich(data, 'education_num')
    
    # hours_per_week, capital_gain, capital_loss
    kdeplot_rich(data, 'hours_per_week')
    kdeplot_rich(data, 'capital_gain')
    kdeplot_rich(data, 'capital_loss')
    
    # 对 workclass 进行处理
    barplotrich(data, 'workclass')
    workclass_dic = {'Self-emp-inc':4, 
                     'Federal-gov':3, 
                     'State-gov':2,'Self-emp-not-inc':2, 'Local-gov':2, 
                     'Private':1, 
                     'Without-pay':0
                     }
    data = feature_dummies(data, 'workclass', workclass_dic)
    
    # 对 education 进行处理
    barplotrich(data, 'education')
    education_dic = {'Preschool':0, 
                     '1st-4th':1, '5th-6th':1, 
                     '7th-8th':2, '9th':2, '10th':2, '11th':2, '12th':2, 
                     'HS-grad':3, 'Some-college':3, 
                     'Assoc-acdm':4, 'Assoc-voc':4, 
                     'Bachelors':5, 
                     'Masters':6, 
                     'Prof-school':7, 'Doctorate':7
                     }
    data['education'] = data['education'].map(education_dic)
    
    # 对 marital_status 进行处理
    barplotrich(data, 'marital_status')
    marital_status_dic = {'Never-married':0, 
                          'Separated':1, 
                          'Married-spouse-absent': 2, \
                          'Divorced':3, 'Widowed':3, 
                          'Married-civ-spouse':4, 'Married-AF-spouse':4 
                          }
    data = feature_dummies(data, 'marital_status', marital_status_dic)
    
    # 对 occupation 进行处理
    barplotrich(data, 'occupation')
    occupation_dic = {'Priv-house-serv':0, 
                      'Other-service':1, 
                      'Handlers-cleaners':2, 
                      'Farming-fishing':3, 'Machine-op-inspct':3, 'Armed-Forces':3, 'Adm-clerical':3, 
                      'Transport-moving':4, 
                      'Craft-repair':5, 
                      'Sales': 6, 
                      'Tech-support':7, 'Protective-serv':7, 
                      'Prof-specialty':8, 'Exec-managerial':8
                      }
    data = feature_dummies(data, 'occupation', occupation_dic)
    
    # 对 relationship 进行处理
    barplotrich(data, 'relationship')
    relationship_dic = {'Own-child':0, 
                        'Other-relative':1,
                        'Unmarried':2, 
                        'Not-in-family':3, 
                        'Wife':4, 'Husband':4
                        }
    data = feature_dummies(data, 'relationship', relationship_dic)
    
    # 对 race 进行处理
    barplotrich(data, 'race')
    race_dic = {'Other':0, 
                'Black':1, 'Amer-Indian-Eskimo':1,
                'White':2, 'Asian-Pac-Islander':2
                }
    data = feature_dummies(data, 'race', race_dic)
    
    # 对 sex 进行处理
    barplotrich(data, 'sex')
    data['sex'] = data['sex'].map({'Male':1, 'Female':0})
    data = pd.get_dummies(data,columns=['sex'],prefix='sex')
    
    # 对 native_country 进行处理
    barplotrich(data, 'native_country')
    native_country_dic = {'Holand-Netherlands':0, 'Outlying-US(Guam-USVI-etc)':0, 
                          'Dominican-Republic':1, 'Columbia':1, 'Guatemala':1, 
                          'Mexico':2, 'Nicaragua':2, 'Peru':2, 'Vietnam':2, 'Honduras':2, 'El-Salvador':2, 'Haiti':2, 
                          'Puerto-Rico':3, 'Trinadad&Tobago':3, 'Laos':3, 'Portugal':3, 'Jamaica':3, 'Ecuador':3, 
                          'Thailand':4, 'Scotland':4, 'Poland':4, 'South':4, 
                          'Ireland':5, 'Hungary':5, 'United-States':5, 'Cuba':5, 'Greece':5, 'China':5, 
                          'Hong':6, 'Philippines':6, 'Canada':6, 'Germany':6, 'England':6, 
                          'Italy':7, 'Yugoslavia':7, 'Cambodia':7, 'Japan':7, 
                          'India':8, 'Iran':8, 'France':8, 
                          'Taiwan':9
                          }
    data = feature_dummies(data, 'native_country', native_country_dic)
    
    # 查看所有特征的相关性系数矩阵图
    corr = data.corr()
    show_corr(corr)
    data = data.drop(['marital_status_4'],axis=1)
    data = data.drop(['education'],axis=1)
    data = data.drop(['fnlwgt'], axis =1)
    print(data.info())
    
    # 过采样
    X = data.drop(['rich'],axis=1)
    x_adasyn, y_adasyn = ADASYN().fit_sample(X, data['rich'])
    y_adasyn = pd.DataFrame(y_adasyn)
    print('poor: ' + str(y_adasyn[y_adasyn[0] ==0].index.size))
    print('rich: ' + str(y_adasyn[y_adasyn[0] ==1].index.size))
    
    # 数据归一化，并且将标准化后的数据转化为dataframe格式
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(x_adasyn)
    
    x_standardized = preprocessing.scale(x_adasyn)
    
    # 利用逻辑回归方法对数据进行分类并测试
    logistic_classification(x_minmax,y_adasyn)
    
    
