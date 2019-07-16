# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:55:22 2019

@author: sf_on
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

columns=['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
         'relationship','race','sex','capital_gain','capital_loss',
         'hours_per_week' ,'native_country','rich']
data = pd.read_csv(open('./adult_data.csv'),header = None, names = columns)

# 查看数据集中是否存在空值
print(data.isnull().sum())

# 去除空值
def deletNull(data):
    # 获取数据集中空值的列，并将其转换为list输出
    i_ = data[data.workclass.isnull()].index.tolist()
    j_ = data[data.occupation.isnull()].index.tolist()
    l_ = data[data.native_country.isnull()].index.tolist()
    # 将多个list合并
    i_ = i_ + j_ + l_
    i_ = list(set(i_))
    i_.sort()
    #将存在空值的行drop掉
    data.drop(i_,axis=0,inplace=True)
deletNull(data)

# 查看数据特征
print(data.describe())
print('poor: ' + str(data[data.rich ==0].index.size))
print('rich: ' + str(data[data.rich ==1].index.size))

# 探索数值型变量 hours_per_week 特征
s = np.sort(data.hours_per_week.unique())
print(data['hours_per_week'][data.rich ==0].describe())
print(data['hours_per_week'][data.rich ==1].describe())

sns.boxplot(x = 'rich', y = 'age',data = data)
plt.show()

#labels = data.groupby('age').size().index
#groups = data.groupby('age')

# 获取离散型变量，每个特征的具体类别和类标号之间的数值关系
# 输入特征的名称，如‘age’输出结果
#    feature    rich_0    rich_1
#    17         328       0
#    18         447       0
#    ......
# 结果作为dataframe输出
def num(varible, data):
#   var = '\''+ varible + '\''
    temp = pd.DataFrame(columns=('rich_0','rich_1'))
    temp_0 = data[data.rich == 0].groupby(varible)
    temp_1 = data[data.rich == 1].groupby(varible)
    temp['rich_0'] = temp_0.size()
    temp['rich_1'] = temp_1.size()
    temp = temp.replace(np.nan,0)
    return temp

# 根据输入的特征，做并列直方图，分析特征和类标号的关联
def pic(varible):
    tick_label = varible.index.tolist()
    #绘制并列柱状图
    x=np.arange(varible.index.size)#柱状图在横坐标上的位置
    bar_width=0.3#设置柱状图的宽度
    plt.subplots(figsize=(20,10))
    plt.bar(x,varible['rich_0'],bar_width,color='salmon',label='rich_0')
    plt.bar(x+bar_width,varible['rich_1'],bar_width,color='orchid',label='rich_1')
    plt.legend()#显示图例，即label
    plt.xticks(x+bar_width/2,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
    plt.show()
    print(varible)    

# 并发作图，同时执行所有离散变量的任务
def show_label():
    num_age = num('age', data)
    pic(num_age)
    num_workclass = num('workclass', data)
    pic(num_workclass)
    num_education = num('education', data)
    pic(num_education)
    num_education_num = num('education_num', data)
    pic(num_education_num)
    num_marital_status = num('marital_status', data)
    pic(num_marital_status)
    num_occupation = num('occupation', data)
    pic(num_occupation)
    num_relationship = num('relationship', data)
    pic(num_relationship)
    num_race = num('race', data)
    pic(num_race)
    num_sex = num('sex', data)
    pic(num_sex)
    num_hours_per_week = num('hours_per_week', data)
    pic(num_hours_per_week)
    num_native_country = num('native_country', data)
    pic(num_native_country)
show_label()

# 将特征传递给x，将类标号传递给y
x = data[['age', 'workclass', 'education','education_num','marital_status','occupation', 'race','sex','hours_per_week', 'native_country']]
y = data[['rich']]
columns = ['workclass', 'education','marital_status', 'occupation', 'race','sex', 'native_country']

# 将标称数据转化为数值数据,不能用于有隐含顺序的标称类数据，会导致数据丢失含义
#    workclass_type = x['workclass'].unique()
#    j = 1
#    for i in workclass_type:
#        x.workclass[x['workclass'] == i] = j
#        j = j+1
# 
def numerization(x, columns):
    for column in columns:
        print(column)
        column_type = x[column].unique()
        j = 1
        for label in column_type:
            x[column][x[column] == label] = j
            j = j + 1
numerization(x, columns)

sns.boxplot(x = 'education', y = 'education_num',data = x)

# 探索数值型变量age特征,并对其进行分箱
s = np.sort(x.age.unique())
x.age.plot(kind='hist', xlim=(15,100),bins = x.age.unique().size)
print(x.age.describe())
bins = [1, 26, 33, 40, 49, 100]
age_bin = pd.cut(x.age, bins)
print(pd.value_counts(age_bin))
labels = [1,2,3,4,5]
x['age'] = pd.cut(x['age'], bins = bins, labels = labels, include_lowest = True)

# 采用adasyn算法进行过采样
from imblearn.over_sampling import ADASYN
#from imblearn.over_sampling import SMOTE
# x_smote, y_smote = SMOTE().fit_sample(x, y)
x_adasyn, y_adasyn = ADASYN().fit_sample(x, y)
x_adasyn = pd.DataFrame(x_adasyn)
x_adasyn.columns = ['age', 'workclass', 'education','education_num','marital_status','occupation', 'race','sex','hours_per_week', 'native_country']
y_adasyn = pd.DataFrame(y_adasyn)
y_adasyn.columns = ['rich']

# 查看离散变量之间以及与类标号之间大的皮尔逊相关性
import seaborn as sns
a = pd.merge(x_adasyn,y_adasyn,left_index=True, right_index=True)
a = a.corr() 
sns.heatmap(a)  
plt.show()
print(a)


# =============================================================================
# # 数据标准化，并且将标准化后的数据转化为dataframe格式
# from sklearn import preprocessing
# x_standardized = preprocessing.scale(x_adasyn)
# x_standardized = pd.DataFrame(x_standardized)
# x_standardized.columns = ['age', 'workclass', 'education','education_num','marital_status','occupation', 'race','sex','hours_per_week', 'native_country']
# =============================================================================

# 利用高斯贝叶斯进行分类
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# 利用十择法查看分类情况
def kfold(x, y):
    preaccnb = []
    f1_nb = []
    num = 1
    model = GaussianNB()
    kf = KFold(n_splits=10)
    for train, test in kf.split(x):
        x_train, x_test = x.loc[train], x.loc[test]
        y_train, y_test = y.loc[train], y.loc[test]
        model.fit(x_train, y_train)
        expected = y_test
        predicted = model.predict(x_test)
        accuracy = accuracy_score(expected, predicted)
        preaccnb.append(accuracy)
        f1 = f1_score(expected, predicted)
        f1_nb.append(f1)
        print("高斯贝叶斯"+str(num)+"测试集准确率:     %s " % accuracy)
        print("高斯贝叶斯"+str(num)+"测试集f1-score:  %s " % f1)
        num = num + 1
kfold(x_adasyn,y_adasyn)

# 划分训练集和测试集
def selection(x,y):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, test_size=0.2,random_state=6)
    return x_train, x_test, y_train, y_test
    

# 利用某方法对数据进行分类并测试
def function_classification(x,y,function=LogisticRegression):
    # 切分数据
    x_train, x_test, y_train, y_test = selection(x,y)
    model = function()
    mf = model.fit(x_train, y_train)
    expected = y_test
    predicted = mf.predict(x_test)
    f1 = f1_score(expected, predicted)
    print(model.score(x_test,y_test))
    print(f1)
    print(metrics.classification_report(expected, predicted)) 
    print(confusion_matrix(expected, predicted, sample_weight=None))
    tn,fp,fn,tp = confusion_matrix(expected, predicted, sample_weight=None).ravel()
    print('tp = ' + str(tp) + ' , fp = ' + str(fp))
    print('fn = ' + str(fn) + ' , tn = ' + str(tn))
    y_score = mf.decision_function(x_test)
    fpr,tpr,threshold = metrics.roc_curve(y_test,y_score)
    roc_auc = metrics.auc(fpr,tpr) ###计算auc的值
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
function_classification(x_adasyn,y_adasyn)

# 导出训练文件和测试文件
def output_csv(x,y):
    x_train, x_test, y_train, y_test = selection(x,y)
    Traindata = pd.merge(x_train,y_train,left_index=True, right_index=True)
    Traindata.to_csv('C://Users/sf_on/Desktop/adult_train.csv',index = False,encoding = 'utf-8')
    Testdata = pd.merge(x_test,y_test,left_index=True, right_index=True)
    Testdata.to_csv('C://Users/sf_on/Desktop/adult_test.csv',index = False,encoding = 'utf-8')
output_csv(x_adasyn,y_adasyn)

