# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:36:18 2019

@author: sf_on
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# 载入kaggle上的数据文件，train有800条记录左右，test有400条记录左右
# kaggle上的数据集的分布并不均匀，实际训练和测试过程中需要重新生成一个数据集
train = pd.read_csv(open('./titanic_train.csv',encoding='UTF-8'),encoding='UTF-8')
test = pd.read_csv(open('./titanic_test.csv',encoding='UTF-8'),encoding='UTF-8')

print(train.info())
print(train.isnull().sum())
print(test.info())
print(test.isnull().sum())

outlier_indices =[]
outlier_list_col_index = pd.DataFrame()
features = ['Age','SibSp','Parch','Fare']

# 查看连续特征的异常值
for col in features:
    Q1 = np.percentile(train[col],25)
    Q3 = np.percentile(train[col],75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outlier_list_col = train[(train[col]<Q1-outlier_step)|(train[col]>Q3+outlier_step)].index
    temp = pd.DataFrame((train[col]<Q1-outlier_step)|(train[col]>Q3+outlier_step), columns = [col])
    outlier_indices.extend(outlier_list_col)
    outlier_list_col_index = pd.concat(objs=[outlier_list_col_index,temp],axis=1)
outlier_indices = Counter(outlier_indices)
multiple_outliers=list(k for k,v in outlier_indices.items() if v > 2)

for feature in features:
    print(train[feature].describe())

for row in multiple_outliers:
    print('line:', row, 'Age:', train.iloc[row,6], 'SibSp:', train.iloc[row,7],
          'Parch:', train.iloc[row,8], 'Fare:', train.iloc[row,10])
print(multiple_outliers)
train = train.drop(multiple_outliers, axis = 0).reset_index(drop=True)

train_len, train_var_num = np.shape(train)
dataset = pd.concat(objs = [train,test],axis=0).reset_index(drop=True)

def show_corr():
    plt.figure(figsize=(10,8))
    g = sns.heatmap(train[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr(),annot=True,fmt = '.2f',cmap = 'coolwarm')
    g.set_xlabel('corr')
    plt.show()
show_corr()

# 特征之间的相关性初探
# ------------------------------------
# 年龄与生存率的关系
def show_Age_Survived():
    g = sns.kdeplot(train['Age'][(train['Age'].notnull())&(train['Survived']==0)],color='Red',shade=True)
    g = sns.kdeplot(train['Age'][(train['Age'].notnull())&(train['Survived']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('Age')
    g.set_ylabel('Frequency')
    g = g.legend(['Not Survived','Survived'])
    plt.show()
show_Age_Survived()

# 性别与生存率的关系
def show_Sex_Survived():
    g = sns.barplot(data=train,x='Sex',y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
show_Sex_Survived()

# 船上兄弟姐妹或配偶数量与生存率的关系
def show_SibSp_Survived():
    g = sns.barplot(data=train,x='SibSp',y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
show_SibSp_Survived()

# 船上父母或子女数量与生存率的关系
def show_Parch_Survived():
    g = sns.barplot(data=train,x='Parch',y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
show_Parch_Survived()

# 船舱等级与生存率的关系
def show_Pclass_Survived():
    g = sns.barplot(data=train,x='Pclass',y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
show_Pclass_Survived()

# 票价与生存率的关系
def show_Fare_Survived():
    plt.figure(figsize=(10,5))
    g = sns.kdeplot(train['Fare'][(train['Fare'].notnull())&(train['Survived']==0)],color='Red',shade=True)
    g = sns.kdeplot(train['Fare'][(train['Fare'].notnull())&(train['Survived']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('Fare')
    g.set_ylabel('Frequency')
    g = g.legend(['Not Survived','Survived'])
    plt.show()
show_Fare_Survived()

#Fare特征的缺失值进行填充
dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())
test['Fare']=test['Fare'].fillna(test['Fare'].median())

#利用柱形图来查看log变换前Fare在整个数据集中的分布
def show_Fare():
    plt.figure(figsize=(10,5))
    g=sns.distplot(train['Fare'],color='M',label='Skewness:%.2f'%(train['Fare'].skew()))
    g.legend(loc='best')
    plt.show()
show_Fare()

# 利用log函数进行数据变换
train['Fare_log'] = train['Fare'].map(lambda i:np.log(i) if i>0 else 0)#map()函数具体将元素进行映射的功能
# 查看变换后的数据分布
def Fare_log_show():
    g = sns.distplot(train['Fare_log'],color='M',label='Skewness:%.2f'%(train['Fare_log'].skew()))
    g.legend(loc='best')
    plt.show()
Fare_log_show()

print(train['Embarked'].describe())
#缺失值填充
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].describe().top)
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].describe().top)
print(train['Embarked'].isnull().sum())
print(dataset['Embarked'].isnull().sum())

# 查看Embarked与生存率的关系
def show_Embarked_Survived():
    g = sns.barplot(data=train,x='Embarked',y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
show_Embarked_Survived()

print(dataset['Name'].head())
print(dataset['FamilyName'].head())

dataset_title = [i.split('.')[0].strip() for i in dataset['FamilyName']]
dataset['Title'] = pd.Series(dataset_title)
title = dataset['Title'].unique()
print(dataset['Title'].describe())
print(title)

for t in title:
    print(str(t) + ':', dataset['Title'][dataset['Title'] == t].size)


# =============================================================================
# Mr.= mister，先生
# Mrs.= mistress，太太/夫人
# Miss,复数为misses，对未婚妇女用
# Ms.或Mz，美国近来用来称呼婚姻状态不明的妇女
# Mme Madame简写是Mme.,复数是mesdames(简写是Mme)
# Mlle,小姐
# Lady, 女士，指成年女子，有些人尤其是长者认为这样说比较礼貌
# Dona，是西班牙语对女子的称谓，相当于英语的 Lady
# Master，佣人对未成年男少主人的称呼,相当于汉语的'少爷'。
# Mr. Mister的略字,相当于汉语中的'先生',是对男性一般的称呼,区别于有头衔的人们,如Doctor, Professor,Colonel等
# Don，n. <西>（置于男士名字前的尊称）先生，堂
# jonkheer 最低的贵族头衔
# Sir 贵族头衔
# Rev.= reverend，用于基督教的牧师，如the Rev. Mr.Smith
# Dr.= doctor，医生/博士
# Capt 船长
# Colonel，上校
# major，意思有少校人意思
# theCountess 女伯爵
# =============================================================================

def title_show():
    plt.figure(figsize=(15,8))
    g = sns.barplot(data = dataset[:train_len],x = 'Title',y = 'Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
title_show()

# =============================================================================
# 第一类：'Mr', 'Don' 普通男性
# 第二类：'Mrs','Miss','Mme','Ms','Lady','Dona','Mlle','theCountess'  女性
# 第三类：'Major','Col','Dr' 军官和医生
# 第四类：'Sir','Master','Jonkheer'  贵族男性
# 第五类：'Rev','Capt'  基本不可能幸存的职业
# =============================================================================

fnames = [['Mr', 'Don'],
          ['Mrs','Miss','Mme','Ms','Lady','Dona','Mlle','theCountess'],
          ['Major','Col','Dr'],
          ['Sir','Master','Jonkheer'],
          ['Rev','Capt']]

dataset['Title_new'] = dataset['Title']
for i in range(len(fnames)):
    dataset['Title_new'][dataset['Title'].isin(fnames[i])] = fnames[i][0]

def Title_new_show():
    plt.figure(figsize=(10,8))
    g = sns.barplot(data = dataset , x = 'Title_new', y = 'Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
Title_new_show()

# 将title转化为哑变量
data = dataset[['Survived','Pclass','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked','Title_new']]
data['Title_new'] = data['Title_new'].map({'Mr':0,'Mrs':1,'Major':2,'Sir':3,'Rev':4})
data['Title_new'] = data['Title_new'].astype(int)
data = pd.get_dummies(data,columns=['Title_new'],prefix='TL')

# 根据另外几种连续特征，对Age进行填充
index_NaN_age = list(data['Age'][data['Age'].isnull()].index)
for i in index_NaN_age:
    age_med = data['Age'].median()
    age_pred = data['Age'][((data['SibSp'] == data.iloc[i]['SibSp']) &
                      (data['Parch'] == data.iloc[i]['Parch']) &
                      (data['Pclass'] == data.iloc[i]['Pclass'])
                      )].median()
    if not np.isnan(age_pred):
        data['Age'].iloc[i] = age_pred
    else:
        data['Age'].iloc[i] = age_med

g=sns.factorplot(data=dataset,x='Survived',y='Age',kind='violin')
plt.show()

def show_Age_Survived(df):
    g = sns.kdeplot(df['Age'][(df['Age'].notnull())&(df['Survived']==0)],color='Red',shade=True)
    g = sns.kdeplot(df['Age'][(df['Age'].notnull())&(df['Survived']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('Age')
    g.set_ylabel('Frequency')
    g = g.legend(['Not Survived','Survived'])
    plt.show()
show_Age_Survived(data)

print(data['Cabin'].describe())
print(data['Cabin'].isnull().sum())
data.drop(labels = ['Cabin'], axis = 1, inplace = True)

# 采用正则将ticket前部分的英文提取，该部分英文有一定的象征含义
import re
Ticket = []
for i in list(data['Ticket']):
    if not i.isdigit():
        Ticket.append(re.sub('[0-9\.\/]', '', i).strip())
    else:
        Ticket.append('X')
data['Ticket'] = Ticket
print(data['Ticket'].describe())

# ticket和生存率的关系
def ticket_survived_show():
    plt.figure(figsize=(12,6))
    g=sns.barplot(data=data,x='Ticket',y='Survived')
    g.set_ylabel('Survival Probability')
    plt.show()
ticket_survived_show()

#性别Sex的数值化
data['Sex']=data['Sex'].map({'male':0,'female':1})
train['Sex']=train['Sex'].map({'male':0,'female':1})

# 将Ticket和Embarked进行哑变量处理
data = pd.get_dummies(data,columns=['Ticket'],prefix='T')
data = pd.get_dummies(data, columns = ['Embarked'], prefix='Em')
data['Pclass'] = data['Pclass'].astype('category')
data = pd.get_dummies(data, columns = ['Pclass'],prefix='Pc')
print(data.info())

from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn import model_selection

# 训练集和测试集准备
X_train, X_test, y_train, y_test = model_selection.train_test_split(
            data.iloc[:,1:], data['Survived'], test_size=0.3,random_state=32)

# 采用十择法和参数自动寻优(网格搜索)找到一个近似的优化参数
RFC = RandomForestClassifier()
rf_param_grid = {'max_depth' : [None],
                 'max_features' : [4, 5, 6],
                 'min_samples_split' : [2, 3, 4],
                 'min_samples_leaf' : [2, 3, 4],
                 'bootstrap' : [False],
                 'n_estimators' : [90, 100, 120],
                 'criterion': ['gini']}
kfold = KFold(n_splits=10)
gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv= kfold, scoring='accuracy', n_jobs= 1, verbose = 1)
gsRFC.fit(X_train,y_train)
print(gsRFC.best_estimator_)
print(gsRFC.best_params_)
print(gsRFC.best_score_ )

# 找出最优化的随机森林参数
rfc_new = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=7, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rfc_new.fit(X_train,y_train)
print(rfc_new.score(X_train,y_train))
print(rfc_new.score(X_test,y_test))

from sklearn.model_selection._validation import learning_curve
# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, 
                        n_jobs = 1, train_size = np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve( \
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes = train_size)
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
plot_learning_curve(rfc_new,'RF_mearning_curves',X_train,y_train,cv=kfold)

# 查找特征的重要性并排序
importances = rfc_new.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(importances)):
    print(X_train.columns.tolist()[f], ':', importances[f])
print(importances[indices])
print(X_train.columns[indices])
feat_labels = X_train.columns.tolist()

# 根据特征重要性进行可视化
def feature_importance_show():
    plt.subplots(figsize = (30,15))
    plt.title('Feature Importance')
    g = sns.barplot(y = X_train.columns[indices],x = importances[indices], orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    plt.show()
feature_importance_show()
    
# 导出训练文件和测试文件
def output_csv():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
            data.iloc[:,1:], data['Survived'], test_size=0.3,random_state=32)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    Traindata = pd.merge(X_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/RandomForest_Titanic_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(X_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/RandomForest_Titanic_test.csv',index = False,encoding = 'utf-8')
output_csv()
    
    
