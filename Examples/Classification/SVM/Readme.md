* [SVM案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM)
  * [SVM原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#1-SVM原理)
  * [HeartDisease案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#2-Heartdisease案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#22-案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#221-数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#222-特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#223-模型训练与评估)
      * [结果可视化](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#224-结果可视化)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KSVM#3-总结)


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



**去除异常值:**


**数据归一化:**


#### 2.2.3 模型训练与评估


#### 2.2.4 结果可视化 
    

## 3 总结

