# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:55:22 2019

@author: sf_on
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from  sklearn import preprocessing

# 工具函数，画相关系数图
def show_corr(x,y):
    cor = pd.merge(x,y,left_index=True, right_index=True)
    cor = cor.corr() 
    plt.figure(figsize=(16,16))
    sns.heatmap(cor)  
    plt.show()

# 工具函数，画Roc曲线
def show_Roc(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

columns=['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
         'relationship','race','sex','capital_gain','capital_loss','hours_per_week' ,'native_country','rich']
data = pd.read_csv(open('./adult_data.csv'),header = None, names = columns)

# 查看数据集中是否存在空值
print(data.isnull().sum())

# 去除空值
# 由于存在空值的列是workclass、occupation、native_country 
# 这三列，都是标称类属性，且与其他列的关联性较弱，不好做关联插值
# workclass 有1836个空值，occupation 1843个空值，native_country 有 583个空值,共计2399个空值
# 缺失样本条数占总数据样本8%左右，影响不大
# 因此，这里我们直接将其去除
def deletNull(data):
    # 获取数据集中空值的列，并将其转换为list输出
    i_ = data[data.workclass.isnull()].index.tolist()
    j_ = data[data.occupation.isnull()].index.tolist()
    l_ = data[data.native_country.isnull()].index.tolist()
    # 将多个list合并
    i_ = i_ + j_ + l_
    i_ = list(set(i_))
    print('data.isnull.length:', len(i_))
    i_.sort()
    #将存在空值的行drop掉
    data.drop(i_,axis=0,inplace=True)
deletNull(data)

# 查看数据特征
# 可以看出样本中，poor和rich的人数比率差别较大，并且由于后续分析主要关注谁会变得富有，因此我们关注比率的指标
print(data.describe())
print('poor: ' + str(data[data.rich ==0].index.size))
print('rich: ' + str(data[data.rich ==1].index.size))


# 数据探索性分析
# ----------------------------------------------------
# age 和 rich
def show_age_rich():
    g = sns.kdeplot(data['age'][(data['rich']==0)],color='Red',shade=True)
    g = sns.kdeplot(data['age'][(data['rich']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('age')
    g.set_ylabel('Frequency')
    g = g.legend(['poor','rich'])
    plt.show()
show_age_rich()

# education_num 和 rich
def show_education_num_rich():
    plt.figure(figsize=(15,4))
    g = sns.barplot(data=data,x='education_num',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_education_num_rich()

# hours_per_week, capital_gain, capital_loss 这三个属性是连续属性
# 可以对这三个属性做离散化
# hours_per_week 和 rich
def show_hours_per_week_rich():
    plt.figure(figsize=(9,4))
    g = sns.kdeplot(data['hours_per_week'][(data['rich']==0)],color='Red',shade=True)
    g = sns.kdeplot(data['hours_per_week'][(data['rich']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('hours_per_week')
    g.set_ylabel('Frequency')
    g = g.legend(['poor','rich'])
    plt.show()
show_hours_per_week_rich()

# capital_gain 和 rich
def show_capital_gain_rich():
    plt.figure(figsize=(9,4))
    g = sns.kdeplot(data['capital_gain'][(data['rich']==0)],color='Red',shade=True)
    g = sns.kdeplot(data['capital_gain'][(data['rich']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('capital_gain')
    g.set_ylabel('Frequency')
    g = g.legend(['poor','rich'])
    plt.show()
show_capital_gain_rich()

# capital_loss 和 rich
def show_capital_loss_rich():
    plt.figure(figsize=(9,4))
    g = sns.kdeplot(data['capital_loss'][(data['rich']==0)],color='Red',shade=True)
    g = sns.kdeplot(data['capital_loss'][(data['rich']==1)],color='Blue',shade=True,ax=g)
    g.set_xlabel('capital_loss')
    g.set_ylabel('Frequency')
    g = g.legend(['poor','rich'])
    plt.show()
show_capital_loss_rich()

# 特征工程
# -----------------------------------------------------------------

x = data[['age','workclass','education','education_num','marital_status','occupation',
         'relationship','race','sex','capital_gain','capital_loss','hours_per_week' ,'native_country']]
y = data[['rich']]

# 对 workclass 进行处理
# workclass 和 rich 的关联关系
def show_workclass_rich():
    plt.figure(figsize=(10,4))
    g = sns.barplot(data=data,x='workclass',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_workclass_rich()

workclass = x.workclass.tolist()
workclass = list(set(workclass))
print(workclass)
x['workclass'] = x['workclass'].map({'Self-emp-inc':4, 'Federal-gov':3, 
 'State-gov':2,'Self-emp-not-inc':2, 'Local-gov':2, 'Private':1, 'Without-pay':0})
x = pd.get_dummies(x,columns=['workclass'],prefix='workclass')

# 对 education 进行处理
# education 和 rich
def show_education_rich():
    plt.figure(figsize=(15,4))
    g = sns.barplot(data=data,x='education',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_education_rich()

education = x.education.tolist()
education = list(set(education))
print(education)
x['education'] = x['education'].map({'Preschool':0, '1st-4th':1, '5th-6th':1, \
                 '7th-8th':2, '9th':2, '10th':2, '11th':2, '12th':2, 'HS-grad':3, \
                 'Some-college':3, 'Assoc-acdm':4, 'Assoc-voc':4, 'Bachelors':5, \
                 'Masters':6, 'Prof-school':7, 'Doctorate':7})

# 查看education和education_num之间以及与类标号之间大的皮尔逊相关性
# 可以看出二者的相关性极高，基本上不同的类型的教学，可以用教学年限来代替
# 存在特征冗余，可以去除education这个特征
show_corr(x,y)
x = x.drop(['education'],axis=1)

# 对 marital_status 进行处理
# marital_status 和 rich
def show_marital_status_rich():
    plt.figure(figsize=(11,4))
    g = sns.barplot(data=data,x='marital_status',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_marital_status_rich()
marital_status = x.marital_status.tolist()
marital_status = list(set(marital_status))
print(marital_status)
x['marital_status'] = x['marital_status'].map({'Never-married':0, 'Separated':1, 'Married-spouse-absent': 2, \
                 'Divorced':3, 'Widowed':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4 })
x = pd.get_dummies(x,columns=['marital_status'],prefix='marital_status')

# 对 occupation 进行处理
# occupation 和 rich
def show_occupation_rich():
    plt.figure(figsize=(22,4))
    g = sns.barplot(data=data,x='occupation',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_occupation_rich()
occupation = x.occupation.tolist()
occupation = list(set(occupation))
print(occupation)
dic_occupation ={'Priv-house-serv':0, 'Other-service':1, 'Handlers-cleaners':2, \
                 'Farming-fishing':3, 'Machine-op-inspct':3, 'Armed-Forces':3, \
                 'Adm-clerical':3, 'Transport-moving':4, 'Craft-repair':5, \
                 'Sales': 6, 'Tech-support':7, 'Protective-serv':7, 'Prof-specialty':8, \
                 'Exec-managerial':8}
x['occupation'] = x['occupation'].map(dic_occupation)
x = pd.get_dummies(x,columns=['occupation'],prefix='occupation')

# 对 relationship 进行处理
# relationship 和 rich
def show_relationship_rich():
    plt.figure(figsize=(8,4))
    g = sns.barplot(data=data,x='relationship',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_relationship_rich()
relationship = x.relationship.tolist()
relationship = list(set(relationship))
print(relationship)
x['relationship'] = x['relationship'].map({'Own-child':0, 'Other-relative':1,'Unmarried':2, \
                     'Not-in-family':3, 'Wife':4, 'Husband':4})
x = pd.get_dummies(x,columns=['relationship'],prefix='relationship')

# 对 race 进行处理
# race 和 rich
def show_race_rich():
    plt.figure(figsize=(9,4))
    g = sns.barplot(data=data,x='race',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_race_rich()
race = x.race.tolist()
race = list(set(race))
print(race)
x['race'] = x['race'].map({'Other':0, 'Black':1, 'Amer-Indian-Eskimo':1,
             'White':2, 'Asian-Pac-Islander':2})
x = pd.get_dummies(x,columns=['race'],prefix='race')

# 对 sex 进行处理
# sex 和 rich
def show_sex_rich():
    g = sns.barplot(data=data,x='sex',y='rich')
    g.set_ylabel('Survival Probability')
    plt.show()
show_sex_rich()
x['sex'] = x['sex'].map({'Male':1, 'Female':0})
x = pd.get_dummies(x,columns=['sex'],prefix='sex')


# 对 native_country 进行处理
# native_country rich比率的顺序
def orderlist():
    datasort = data[['native_country','rich']]
    grouped = datasort.groupby('native_country')
    sort_list = []
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

# native_country 和 rich 的可视化
def show_native_country_rich():
    order = orderlist()   
    plt.figure(figsize=(50,4))
    g = sns.barplot(data = data, x='native_country',y='rich',order=order)
    g.set_ylabel('Survival Probability')
    plt.show()
    
show_native_country_rich()
native_country = x.native_country.tolist()
native_country = list(set(native_country))
print(native_country)
dic_native_country = {'Holand-Netherlands':0, 'Outlying-US(Guam-USVI-etc)':0, 
                      'Dominican-Republic':1, 'Columbia':1, 'Guatemala':1, 
                      'Mexico':2, 'Nicaragua':2, 'Peru':2, 'Vietnam':2, 'Honduras':2, 'El-Salvador':2, 'Haiti':2, 
                      'Puerto-Rico':3, 'Trinadad&Tobago':3, 'Laos':3, 'Portugal':3, 'Jamaica':3, 'Ecuador':3, 
                      'Thailand':4, 'Scotland':4, 'Poland':4, 'South':4, 
                      'Ireland':5, 'Hungary':5, 'United-States':5, 'Cuba':5, 'Greece':5, 'China':5, 
                      'Hong':6, 'Philippines':6, 'Canada':6, 'Germany':6, 'England':6, 
                      'Italy':7, 'Yugoslavia':7, 'Cambodia':7, 'Japan':7, 
                      'India':8, 'Iran':8, 'France':8, 
                      'Taiwan':9}
x['native_country'] = x['native_country'].map(dic_native_country)
x = pd.get_dummies(x,columns=['native_country'],prefix='native_country')

# 查看所有特征的相关性系数矩阵图
# 其中relationship_4 和 marital_status_4 高度正相关，经分析，二者均代表已婚夫妻的含义，因此可以去除一组
# race_1,race_2 以及 sex_0,sex_1 还有workclass1,workclss2 高度负相关，这是由于哑变量后造成的，因此保留这些特征
# 其余特征都予以保留
show_corr(x,y)
x = x.drop(['marital_status_4'],axis=1)

# 过采样
x_adasyn, y_adasyn = ADASYN().fit_sample(x, y)
x_adasyn = pd.DataFrame(x_adasyn)
y_adasyn = pd.DataFrame(y_adasyn)
print('poor: ' + str(y_adasyn[y_adasyn[0] ==0].index.size))
print('rich: ' + str(y_adasyn[y_adasyn[0] ==1].index.size))

# 数据归一化，并且将标准化后的数据转化为dataframe格式
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x_adasyn)
x_minmax = pd.DataFrame(x_minmax)
# 数据标准化
x_standardized = preprocessing.scale(x_adasyn)
x_standardized = pd.DataFrame(x_standardized)

# 利用逻辑回归方法对数据进行分类并测试
def function_classification(x,y):
    # 切分数据
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y,test_size=0.3,random_state=36)
    
    model = LogisticRegression()
    mf = model.fit(x_train, y_train)
    expected = y_test
    predicted = mf.predict(x_test)
    f1 = f1_score(expected, predicted)
    
    print('accuracy_score：', model.score(x_test,y_test))
    print('f1_score：', f1)
    print(metrics.classification_report(expected, predicted)) 
    
    # 画Roc曲线
    tn,fp,fn,tp = confusion_matrix(expected, predicted, sample_weight=None).ravel()
    y_score = mf.decision_function(x_test)
    fpr,tpr,threshold = metrics.roc_curve(y_test,y_score)
    roc_auc = metrics.auc(fpr,tpr) #计算auc的值
    show_Roc(fpr,tpr,roc_auc)

function_classification(x_minmax,y_adasyn)

# 导出训练文件和测试文件
def output_csv(x,y):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,test_size=0.3,random_state=36)
    Traindata = pd.merge(x_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/adult_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(x_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/adult_test.csv',index = False,encoding = 'utf-8')
output_csv(x_adasyn,y_adasyn)

