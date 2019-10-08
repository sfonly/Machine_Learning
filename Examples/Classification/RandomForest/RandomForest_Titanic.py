# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:36:18 2019

RandomForest 随机森林

@author: sf_on
"""

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection._validation import learning_curve
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn import model_selection


def find_outliers(data, features, n = 1):
    '''
    寻找异常值
    Parameters:
        data:           数据集
        features:       需要判断存在异常值的特征
        n:              存在 n 个异常特征, 即为异常值
    Return:
        multiple_outliers:  异常值的 index 
    '''
    outlier_indices = [] # 存储异常值的index

    for col in features:
        Q1 = np.percentile(train[col],25)
        Q3 = np.percentile(train[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = data[(data[col]<Q1-outlier_step)|(data[col]>Q3+outlier_step)].index
        outlier_indices.extend(outlier_list_col)
            
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k,v in outlier_indices.items() if v >= n)
    
    return multiple_outliers


def handle_outliers(data, continues_features, k = 1):
    '''
    处理异常值
    打印要删除的列，并返回一个 drop 过后的对象
    Parameters:
        data:                   数据集
        continues_features:     需要判断存在异常值的连续特征
        n:                      存在 n 个异常特征, 即为异常值
    Return：
        data：                   去除异常值后的数据集
    '''
    multiple_outliers = find_outliers(dataSet, continues_features,k)    

    for row in multiple_outliers:           
        print('line: %s, Age: %s, SibSp: %s, Parch: %s, Fare: %s' \
              %(row,data['Age'][row],data['SibSp'][row],data['Parch'][row],data['Fare'][row]))

    data = data.drop(multiple_outliers, axis = 0).reset_index(drop=True)
    return data


def show_corr(data):
    '''
    协方差矩阵图
    Parameters:
        data:                   数据集
    '''
    plt.figure(figsize=(10,8))
    g = sns.heatmap(data.corr(),annot=True,fmt = '.2f',cmap = 'coolwarm')
    g.set_xlabel('corr')
    plt.show()


def barplot_Survived(data, col):
    '''
    柱状图
    Parameters:
        data:       数据集
        col:        特征
    '''
    if len(data[col].unique()) <= 8:
        plt.figure(figsize=(8,5))
    else:
        plt.figure(figsize=(15,5))
    
    g = sns.barplot(data = data,x = col,y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()


def kdeplot_Survived(data, col):
    '''
    曲线图
    Parameters:
        data:       数据集
        col:        特征
    '''
    plt.figure(figsize=(8,5))        
    g = sns.kdeplot(data[col][(data[col].notnull())&(data['Survived']==0)],color='Red',shade=True)
    g = sns.kdeplot(data[col][(data[col].notnull())&(data['Survived']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel(col)
    g.set_ylabel('Frequency')
    g = g.legend(['Not Survived','Survived'])
    plt.show()


def distplot(data,col):
    '''
    柱形曲线叠加图
    Parameters:
        data:       数据集
        col:        特征
    '''
    plt.figure(figsize=(8,5))
    g=sns.distplot(data[col],color='Blue',label='Skewness:%.2f'%(data[col].skew()))
    g.legend(loc='best')
    plt.show()


def preprocesing_Firstname(dataSet):
    '''
    处理 Firstname ,提取fname的前缀，并将其分为五类存在 Title_new 中
    Parameters:
        dataSet:       数据集
    '''
    fnames = [['Mr', 'Don'],
              ['Mrs','Miss','Mme','Ms','Lady','Dona','Mlle','theCountess'],
              ['Major','Col','Dr'],
              ['Sir','Master','Jonkheer'],
              ['Rev','Capt']]
    
    dataSet['Title'] = [i.split('.')[0].strip() for i in dataSet['FamilyName']]
    title = dataSet['Title'].unique()
    
    for t in title:
        print(str(t) + ':', dataSet['Title'][dataSet['Title'] == t].size)
    barplot_Survived(dataSet, 'Title')
    
    dataSet['Title_new'] = dataSet['Title']
    for i in range(len(fnames)):
        dataSet['Title_new'][dataSet['Title'].isin(fnames[i])] = fnames[i][0]


def preprocesing_Ticket(data):
    '''
    采用正则公式提取 ticket 前面的部分英文
    Parameters:
        data:       数据集
    '''
    Ticket = []
    for i in list(data['Ticket']):
        if not i.isdigit():
            Ticket.append(re.sub('[0-9\.\/]', '', i).strip())
        else:
            Ticket.append('X')
    data['Ticket_new'] = Ticket


def preprocesing_Age(dataSet):
    '''
    根据另外几种连续特征，填充 Age
    Parameters:
        dataSet:       数据集
    '''
    index_NaN_age = list(dataSet['Age'][dataSet['Age'].isnull()].index)
    for i in index_NaN_age:
        age_med = dataSet['Age'].median()
        age_pred = dataSet['Age'][((dataSet['SibSp'] == dataSet.iloc[i]['SibSp']) &
                          (dataSet['Parch'] == dataSet.iloc[i]['Parch']) &
                          (dataSet['Pclass'] == dataSet.iloc[i]['Pclass'])
                          )].median()
        if not np.isnan(age_pred):
            dataSet['Age'].iloc[i] = age_pred
        else:
            dataSet['Age'].iloc[i] = age_med


def factorplot_Survived(data, col):
    '''
    分布图
    Parameters:
        data:       数据集
        col：       特征
    '''
    sns.factorplot(data=data,x='Survived',y=col,kind='violin')
    plt.show()


def Kfold_RF(X_train, X_test, y_train, y_test, param_grid, k = 10):
    '''
    采用十择法和参数自动寻优(网格搜索)找到一个近似的优化参数
    Parameters:
        X_train, X_test:        数据集
        y_train, y_test：       特征
        param_grid：            参数网格 
        k：                     十择法默认为 10
    '''
    RFC = RandomForestClassifier()
    kfold = KFold(n_splits = k)
    gsRFC = GridSearchCV(RFC, param_grid = param_grid, cv= kfold, scoring='accuracy', n_jobs= 1, verbose = 1)
    gsRFC.fit(X_train,y_train)
    
    print('----------------------------------------')
    print('best_estimator_: ', gsRFC.best_estimator_)
    print('----------------------------------------')
    print('best_params_: ', gsRFC.best_params_)
    print('----------------------------------------')
    print('best_score_: ', gsRFC.best_score_ )
    print('----------------------------------------')


def plot_learning_curve(estimator, X, y, ylim = None, k = 5):
    '''
    学习曲线
    Parameters:
        estimator：          训练后的模型
        X：                  特征
        y：                  类标号
        ylim：               y坐标轴
        k         训练集的切割数目
    '''
    train_size = np.linspace(.1, 1.0, k)
    plt.figure(figsize=(10,6))
    plt.title('RF_learning_curves')
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)
    
    train_sizes, train_scores, test_scores = learning_curve( \
            estimator, X, y, cv=None, n_jobs = 1, train_sizes = train_size)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")    
    plt.legend(loc="best")
    plt.show()   


def importance_feature(data, estimator, k):
    '''
    查看前 k 个最重要的特征, 并可视化
    Parameters：
        data：           数据集
        estimator：      模型
        k:               前 k 个重要的特征       
    '''
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = data.columns[indices][:k]
    importance_value = importances[indices][:k]
    
    for n in range(len(indices)):
        print('%s : %s' %(data.columns[indices][n],importances[indices][n]))

    plt.subplots(figsize = (10,12))
    plt.title('Feature Importance')
    g = sns.barplot(y = features,x = importance_value, orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    plt.show()


if __name__ == '__main__':    

    train = pd.read_csv(open('./titanic_train.csv',encoding='UTF-8'),encoding='UTF-8')
    test = pd.read_csv(open('./titanic_test.csv',encoding='UTF-8'),encoding='UTF-8')
    
    # 查看kaggle上数据集的特征
    print(train.info())
    print(test.info())
    
    # 将二者合并为一个数据集
    dataSet = pd.concat(objs = [train,test],axis=0).reset_index(drop=True)
    print(dataSet.info())
    print(dataSet.isnull().sum())
    
    print('Survived: ', dataSet[dataSet['Survived'] == 1].shape[0])
    print('Not Survived:',dataSet[dataSet['Survived'] == 0].shape[0])
    
    # 查看数据集中连续特征的情况
    continues_features = ['Age','SibSp','Parch','Fare']
    for feature in continues_features:
        print(dataSet[feature].describe())
        print('%s is null: %s' %(feature,dataSet[feature].isnull().sum()))
    
    # 处理特异点，以及查看特征相关性
    dataSet = handle_outliers(dataSet, continues_features, k = 3)
    show_corr(dataSet)
    print(np.shape(dataSet))
    
    # ----------------------------------
    # 数据探索
    # 性别、船上兄弟姐妹或配偶数量、船上父母或子女数量、船舱等级与生存率的关系
    cols = ['Sex','SibSp', 'Parch', 'Pclass']
    for col in cols:
        barplot_Survived(dataSet, col)
    
    # Fare与生存率的关系
    # Fare特征的缺失值进行填充, 并利用log函数进行数据变换
    kdeplot_Survived(dataSet, 'Fare')
    dataSet['Fare'] = dataSet['Fare'].fillna(dataSet['Fare'].median())
    dataSet['Fare_log'] = dataSet['Fare'].map(lambda i:np.log(i) if i>0 else 0) #map()函数具体将元素进行映射的功能
    distplot(dataSet, 'Fare')
    distplot(dataSet, 'Fare_log')
    
    # 查看Embarked与生存率的关系
    dataSet['Embarked'] = dataSet['Embarked'].fillna(dataSet['Embarked'].describe().top)
    barplot_Survived(dataSet, 'Embarked')
    
    # FamilyName
    print(dataSet['FamilyName'].head())

    # 查看处理过后的Title_name, 并用其代替FamilyName
    preprocesing_Firstname(dataSet)
    barplot_Survived(dataSet,'Title_new')
    
    # 年龄与生存率的关系    
    print('before preprocesing_Age: ')
    kdeplot_Survived(dataSet, 'Age')
    factorplot_Survived(dataSet,'Age')
    preprocesing_Age(dataSet)
    
    print('after preprocesing_Age: ')
    kdeplot_Survived(dataSet, 'Age')
    factorplot_Survived(dataSet,'Age')
    
    # Cabin
    print(dataSet['Cabin'].describe())
    print(dataSet['Cabin'].isnull().sum())
    dataSet.drop(labels = ['Cabin'], axis = 1, inplace = True)
    
    # ticket和生存率的关系
    preprocesing_Ticket(dataSet)
    barplot_Survived(dataSet,'Ticket_new')
    print(dataSet['Ticket_new'].describe())
    
    # --------------------------
    # 数据预处理
    data = dataSet[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare_log',
                    'Embarked','Title_new','Ticket_new']]
    
    # 将title转化为哑变量
    data['Title_new'] = data['Title_new'].map({'Mr':0,'Mrs':1,'Major':2,'Sir':3,'Rev':4})
    data['Title_new'] = data['Title_new'].astype(int)
    data = pd.get_dummies(data,columns=['Title_new'],prefix='TL')
    
    #性别Sex的数值化
    data['Sex'] = data['Sex'].map({'male':0,'female':1})
    
    # 将Ticket,Embarked,Pclass进行哑变量处理
    data = pd.get_dummies(data,columns=['Ticket_new'],prefix='T')
    data = pd.get_dummies(data, columns = ['Embarked'], prefix='Em')
    data = pd.get_dummies(data, columns = ['Pclass'],prefix='Pc')
    
    print(data.info())
    
    # ------------------------------
    # 训练模型
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                data.iloc[:,1:], data['Survived'], test_size=0.3,random_state=32)
    
    # 设置随机森林模型参数网格
    rf_param_grid = {'max_depth' : [None],
                     'max_features' : [4, 5, 6, 7, 8],
                     'min_samples_split' : [2, 3, 4],
                     'min_samples_leaf' : [2, 3, 4],
                     'bootstrap' : [False],
                     'n_estimators' : [70, 80, 90, 100, 110],
                     'criterion': ['gini']}
    
    Kfold_RF(X_train, X_test, y_train, y_test, rf_param_grid)
    
    # 找出最优化的随机森林参数
    rfc_new = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                max_depth=None, max_features=7, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=3, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    rfc_new.fit(X_train,y_train)
    
    print('rfc_new.score(train):', rfc_new.score(X_train,y_train))
    print('rfc_new.score(test):', rfc_new.score(X_test,y_test))
    
    plot_learning_curve(rfc_new,X_train,y_train) # 绘制学习曲线
    importance_feature(X_train, rfc_new, 30) # 查看最重要的 k 个特征
