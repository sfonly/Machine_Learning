* [随机森林案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest)
  * [随机森林原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#1-随机森林原理)
  * [Titanic案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#2-Titanic案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#22-案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#221-数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#222-特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#223-模型训练与评估)
      * [结果可视化](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#224-结果可视化)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/RandomForest#3-总结)


# 随机森林案例学习笔记
## 1 随机森林原理
**随机森林定义：**  
[随机森林-维基百科](https://zh.wikipedia.org/wiki/随机森林)



**优点：**  


**缺点：**  


## 2 Titanic案例
### 2.1 案例背景




**数据集描述:**

|   |  feature_name  | feature_type | structure | describe             |
| - | :----:         | :----:       | :----:    | :----:               |
| 0 | PassengerID    | continues    | int       | 乘客ID               |
| 1 | Survival       | discrete     | int       | 是否存活              |
| 2 | Pclass         | continues    | norminal  | 船舱登记              |
| 3 | Name           | discrete     | norminal  | 名                   |
| 4 | FirstName      | discrete     | norminal  | 姓                   |
| 5 | Sex            | discrete     | norminal  | 性别                 |
| 6 | Age            | continues    | int       | 年龄                 |
| 7 | SibSp          | continues    | int       | 船上兄弟姐妹/配偶的数量|
| 8 | Parch          | discrete     | norminal  | 船上父母子女的数量     |
| 9 | Ticket         | discrete     | norminal  | 船票名称              |
|10 | Fare           | continues    | float     | 船票的价格            |
|11 | Cabin          | discrete     | norminal  | 客舱号码              |
|12 | Embarked       | discrete     | norminal  | 上传港口              |


**预测值描述:**
``` python
print('Survived:', dataSet[dataSet['Survived'] == 1].shape[0])
print('Not Survived:',dataSet[dataSet['Survived'] == 0].shape[0])
Survived: 490
Not Survived: 806
```

### 2.2 案例实验

#### 2.2.1 数据预处理


**数据集整体描述分析:**
``` python
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 13 columns):
PassengerId    1309 non-null int64
Survived       1309 non-null int64
Pclass         1309 non-null int64
Name           1309 non-null object
FamilyName     1309 non-null object
Sex            1309 non-null object
Age            1046 non-null float64
SibSp          1309 non-null int64
Parch          1309 non-null int64
Ticket         1309 non-null object
Fare           1308 non-null float64
Cabin          295 non-null object
Embarked       1307 non-null object
dtypes: float64(2), int64(5), object(6)
memory usage: 133.0+ KB
None
PassengerId       0
Survived          0
Pclass            0
Name              0
FamilyName        0
Sex               0
Age             263
SibSp             0
Parch             0
Ticket            0
Fare              1
Cabin          1014
Embarked          2
dtype: int64
```

**去除异常值:**  

``` python
line: 27, Age: 19.0, SibSp: 3, Parch: 2, Fare: 263.0
line: 88, Age: 23.0, SibSp: 3, Parch: 2, Fare: 263.0
line: 159, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 180, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 201, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 324, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 341, Age: 24.0, SibSp: 3, Parch: 2, Fare: 263.0
line: 792, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 846, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 863, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 944, Age: 28.0, SibSp: 3, Parch: 2, Fare: 263.0
line: 1079, Age: nan, SibSp: 8, Parch: 2, Fare: 69.55
line: 1251, Age: 14.5, SibSp: 8, Parch: 2, Fare: 69.55
```

#### 2.2.2 特征工程

**特征相关性分析:**

![loss](./pictures/corr.jpg)
    
**特征分析:**

![loss](./pictures/Sex_barplot.png)

![loss](./pictures/SibSp_barplot.png)

![loss](./pictures/Parch_barplot.png)

![loss](./pictures/Pclass_barplot.png)

![loss](./pictures/Fare_kdeplot.png)

![loss](./pictures/Fare_distplot.png)

![loss](./pictures/Fare_Log_distplot.png)

![loss](./pictures/Embarked_barplot.png)

![loss](./pictures/Title_barplot.png)

![loss](./pictures/Title_new_barplot.png)

![loss](./pictures/Age_kdeplot.png)

![loss](./pictures/Age_factorplot.png)

![loss](./pictures/Age_new_kdeplot.png)

![loss](./pictures/Age_new_factorolot.png)

![loss](./pictures/Title_new_barplot.png)






**数据标准化:**


#### 2.2.3 模型训练与评估

**利用网格参数和十择法寻找最优化参数:**  

    设置网格参数，以及对于的方法封装  
``` python
# 设置随机森林模型参数网格
rf_param_grid = {'max_depth' : [None],
                  'max_features' : [4, 5, 6, 7, 8],
                  'min_samples_split' : [2, 3, 4],
                  'min_samples_leaf' : [2, 3, 4],
                  'bootstrap' : [False],
                  'n_estimators' : [70, 80, 90, 100, 110],
                  'criterion': ['gini']}

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
```

    结果
``` python
----------------------------------------
best_estimator_:  RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=5, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
----------------------------------------
best_params_:  {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 5, 'min_samples_leaf': 4, 'min_samples_split': 3, 'n_estimators': 70}
----------------------------------------
best_score_:  0.8743109151047409
----------------------------------------
```

**随机森林建模:**  
``` python
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

rfc_new.score(train): 0.8930540242557883
rfc_new.score(test): 0.8534704370179949
```
#### 2.2.4 结果可视化 

**学习曲线:**  
![loss](./pictures/RF_learning_curves.png)

**特征重要性排名:**  
![loss](./pictures/Features_Importance.png)


## 3 总结


   