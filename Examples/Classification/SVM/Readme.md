* [SVM案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM)
  * [SVM原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#1-SVM原理)
  * [HeartDisease案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#2-Heartdisease案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#22-案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#221-数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#222-特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#223-模型训练与评估)
      * [结果可视化](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#224-结果可视化)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#3-总结)


# SVM案例学习笔记
## 1 SVM原理
**SVM定义：**  
[支持向量机-维基百科](https://zh.wikipedia.org/wiki/支持向量机)

    我的理解：
    支持向量机是通过在高维空间中，构建最大间隔的超平面，从而实现分类或回归的任务

**优点：**  
1. 有完美的理论基础和数学证明
2. 稀疏性，即少量样本就可以获得较好的分类效果
3. 泛化能力好，有较强的鲁棒性
4. 避免矩阵运算的“维度灾难”,计算的复杂性取决于支持向量的数目,而不是样本空间的维数
5. 可以用于解决非线性问题
6. 训练速度较快,SVM 的最终决策函数只由少数的支持向量所确定

**缺点：**  
1. 由于 SVM 算法在二次规划求解参数时，需要大量的内存空间，因而大规模训练样本难以实施
2. 用 SVM 解决多分类问题存在困难
3. 高维空间中核函数的表达存在困难
4. 对缺失数据表现敏感
5. 解出的模型的参数很难理解

## 2 Heartdisease案例
### 2.1 案例背景

    心血管疾病，又称“心脏病”，已逐渐成为威胁人类健康的“第一杀手”，是目前致死最多的疾病，它的危害无年龄、身份、地域之分。
    每年，全世界约有1750万人死于心脏病和其并发症，约占全部死亡人数的30%；
    幸存者超过2000万，但其中大多数人依然有很高的复发和死亡风险。

    在中国，每年大约有260万人死于心脑血管疾病，死亡人数位列世界第二。
    中国心脏病高发已成趋势，每年新发50万人，现患200万人，而每年接受“搭桥和介入”等心脏病治疗的患者连12万人都不到。
    据世界卫生组织统计，到2020年，中国每年因心血管疾病死亡的人数将可能达到400万。
    一项针对国人的调查表明，在高度重视心脏健康的98%人群中，有63%的人已被诊断出至少有一种影响心血管健康的“医学问题”。

    不良生活方式或习惯是导致心血管病危险因素的主要原因，心血管疾病本身是可防可控的。
    如果能改善其危险因素，如：高血压、高血糖、摄入不足、超重肥胖、体力活动过少等情况，心脏病的发生将被有效遏制。
    发达国家心脏病率和死亡率明显下降，其主要原因就是强调了预防。
    
    根据美国某区域的心脏病患者的体侧情况，我们采集了303个用户的体征数据，这些特征是目前心血管医科判定心脏病的主要因素。
    我们针对四种不同的心血管疾病进行分析，这些心血管疾病可能会出现在同一个人身上。
    利用数据挖掘的方式，帮助医生从人们的体测数据中，快速的区分正常人和病人。


**数据集描述:**

|      |feature_name      | feature_type | structure | describe            |
| ---- | :----:           | :----:       | :----:    | :----:              |
| 0 | age                 | continues    | float     | 年龄                |
| 1 | sex                 | discrete     | norminal  | 性别                |
| 2 | chest_pain          | discrete     | norminal  | 是否胸痛             |
| 3 | blood pressure      | continues    | float     | 血压                |
| 4 | serum_cholestoral   | continues    | float     | 血液中胆固醇含量     |
| 5 | fasting_blood_sugar | discrete     | norminal  | 空腹血糖             |
| 6 | electrocardiographic| discrete     | norminal  | 心电图结果           |
| 7 | max_heart_rate      | continues    | int       | 最大心跳数           |
| 8 | induced_angina      | discrete     | norminal  | 运动心绞痛           |
| 9 | ST_depression       | continues    | float     | ST段压力数值（心电图）|
|10 | slope               | discrete     | norminal  | ST倾斜度（心电图）    |
|11 | vessels             | discrete     | norminal  | 透视看到的血管数      |
|12 | thal                | discrete     | norminal  | 缺陷类型             |
|13 | diagnosis           | discrete     | norminal  | 心血管疾病类型        |



**预测值描述:**
``` python
print(data[data['diagnosis'] == 0].shape[0])
print(data[data['diagnosis'] == 1].shape[0])
print(data[data['diagnosis'] == 2].shape[0])
print(data[data['diagnosis'] == 3].shape[0])
164
55
36
35

print(data[data['diagnosis'] == 0].shape[0])
print(data[data['diagnosis'] >= 1].shape[0])
164
139
```

### 2.2 案例实验

#### 2.2.1 数据预处理
    
**去除空值:**
``` python
print(data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
age                     303 non-null float64
sex                     303 non-null float64
chest_pain              303 non-null float64
blood pressure          303 non-null float64
serum_cholestoral       303 non-null float64
fasting_blood_sugar     303 non-null float64
electrocardiographic    303 non-null float64
max_heart_rate          303 non-null float64
induced_angina          303 non-null float64
ST_depression           303 non-null float64
slope                   303 non-null float64
vessels                 303 non-null object
thal                    303 non-null object
diagnosis               303 non-null int64
```
    
实际上，在该数据集中，空值由 ‘？’ 代替，因此，需要对这种特殊字符进行处理
这里，由于缺失的维度为离散值，且只有6个缺失值，因此用众数进行填充
``` python
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
```

#### 2.2.2 特征工程

**特征相关性分析:**

    根据皮尔逊相关系数分析，特征之间的相关性在 (-0.5, 0.5) 区间内，可以全部保留

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/SVM/corr.jpg" width = 60% height = 60% />

**连续特征分析:**

    通过绘制联系特征的散点矩阵图，可以发现这些联系特征呈现非线性的情况，判断正常人和患病人员很难通过单一特征进行判别
    
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/SVM/continues_features.jpg" width = 60% height = 60% />

**数据标准化:**

    因为特征间的差异巨大，加上之前分析连续特征时，这些特征大多呈现高斯分布
    因此，这里对特征统一进行标准化处理

#### 2.2.3 模型训练与评估
``` python
model = SVC(C=1.0, kernel= kernel, gamma='auto_deprecated',
            shrinking=True, probability=False,
            tol=1e-3, cache_size=200, class_weight=None,
            verbose=False, max_iter=-1, decision_function_shape='ovr',
            random_state=None)
```
    这里的kernel分别测试了线性核、高斯核、多项式核
    最后发现，在当前数据集下，线性核的表现最好。
    
    sklearn具体的参数可以参考
[SVC-sklearn官网](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

#### 2.2.4 结果可视化 
```python
accuracy score:  0.87
f1 score:  0.8433734939759038
tp = 35 , fp = 2
fn = 11 , tn = 52
              precision    recall  f1-score   support

           0       0.83      0.96      0.89        54
           1       0.95      0.76      0.84        46

   micro avg       0.87      0.87      0.87       100
   macro avg       0.89      0.86      0.87       100
weighted avg       0.88      0.87      0.87       100
```

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/SVM/confusion_matrix.jpg" width = 40% height = 40% />

     这里评价的指标，首先是整个模型的准确率，在0.87左右（这个指标在本实验中，其实并不是特别重要）
     主要的原因是，我们要做的是找出可能患病的人群，而不是找出没有患病的人群
     因此，这里最重要的评价指标是tp和fn，即判断为 1 是的准确率和召回率
     
     我们希望准确率尽可能的大，并且召回率尽可能的小
     准确率大，说明我们只要判断时病人，那么我们就没有发生误判
     召回率高，就说明fn值小，我们没有遗漏，即不存在没有查出并的情况
     
     最后实际是准确率达到了0.95，还算是不错的指标。
     但是召回率就不是很让人满意，这可能和数据样本数量太少有关

## 3 总结

    SVM 的参数其实并不多，由于支持向量是计算出来的，如果支持向量没有变化，最后的模型就没有变化
    在 Heartdisease 案例中，由于数据样本较少，调参过后，支持向量没有变化，最后模型就没有变化
    但是并不代表 SVM 不需要调参，实际上核函数的选择、惩罚系数影响巨大
    选择不同的核函数，还需要去调整不同核函数的参数，都会对模型的判断结果有影响

  **SVM 的证明实在是太巧妙了，我画了很长的时间去理解和思考它背后的数学原理和推导过程，以及如何利用数学工具去解决和转化问题**
   
