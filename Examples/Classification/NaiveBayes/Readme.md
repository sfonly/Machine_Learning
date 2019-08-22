* [朴素贝叶斯案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes)
  * [NaiveBayes原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#1-NaiveBayes原理)
  * [wine案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#2-Wine案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#22-案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#221-数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#222-特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#223-模型训练与评估)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#3-总结)


# 朴素贝叶斯案例学习笔记
## 1 NaiveBayes原理
**NaiveBayes定义：**  
[朴素贝叶斯分类器-维基百科](https://zh.wikipedia.org/wiki/朴素贝叶斯分类器)

    朴素贝叶斯是假定各种特征间朴素（不相关）的情况下，利用条件概率进行分类的简单概率分类器

**优点：**  
1. 朴素贝叶斯来源于古典数学，有坚实的理论基础，分类效果稳定可靠
2. 对较小规模的数据集表现良好
3. 能处理多分类任务，适合增量训练
4. 对缺失数据不太敏感，算法也比较简单
5. 模型可解释度非常高

**缺点：**  
1. 朴素贝叶斯必须假定各特征间条件无关，而实际工作中，特征间并不是无关的
2. 需要计算先验概率
3. 分类决策存在一定的错误率
4. 对输入数据的表现形式敏感
5. 只能用于分类问题

## 2 Wine案例
### 2.1 案例背景
    
    红酒的品质、品种与多种因素息息相关，通过控制红酒的重要指标，可以极大的提升红酒的品质。
    然而，就算是同一地区、同一品种的葡萄，生产的红酒可能也会天差地别。
    
    是什么影响了红酒的品质？又是哪些因素决定了一种红酒的品牌？怎样去酿造我们理想中更丰富口味的红酒？
    在本案例中，让我们分析红酒数据来探讨一下这些问题。
    
    该实验的数据集是MostPopular Data Sets（hits since 2007）中的wine数据集。
    这是是对在意大利同一地区生产的三种不同品种的酒，做大量化学分析所得出的数据。
    这些数据包括了三种酒中13种不同成分的数量，而这13种不同的特征使得3种品牌的红酒具有不一样的风味。
    那么，能否反过来，通过这13种特征的表现，快速的品鉴出某红酒属于哪一个品牌？

**数据集描述:**

|      |feature_name               | feature_type | structure | describe            |
| ---- | :----:                    | :----:       | :----:    | :----:              |
| 0 | category                     | norminal     | norminal  | 类别                 |
| 1 | Alcohol                      | continues    | float     | 酒精                 |
| 2 | Malic acid                   | continues    | float     | 苹果酸               |
| 3 | Ash                          | continues    | float     | 灰分                 |
| 4 | Alcalinity of ash            | continues    | float     | 灰分碱性             |
| 5 | Magnesium                    | continues    | float     | 镁                   |
| 6 | Total phenols                | continues    | float     | 总酚                 |
| 7 | Flavanoid                    | continues    | float     | 黄酮素类              |
| 8 | Nonflavanoid phenols         | continues    | float     | 非黄酮类酚类          |
| 9 | Proanthocyanins              | continues    | float     | 原花青素              |
|10 | Color intensity              | continues    | float     | 颜色强度              |
|11 | Hue                          | continues    | float     | 色调                  |
|12 | OD280/OD315 of diluted wines | continues    | float     | 稀释葡萄酒OD280/OD315 |
|13 | Proline                      | continues    | float     | 脯氨酸                |

**预测值描述:**

``` python
print(data.info())
Data columns (total 14 columns):
category                        178 non-null int64
Alcohol                         178 non-null float64
Malic acid                      178 non-null float64
Ash                             178 non-null float64
Alcalinity of ash               178 non-null float64
Magnesium                       178 non-null int64
Total phenols                   178 non-null float64
Flavanoid                       178 non-null float64
Nonflavanoid phenols            178 non-null float64
Proanthocyanins                 178 non-null float64
Color intensity                 178 non-null float64
Hue                             178 non-null float64
OD280/OD315 of diluted wines    178 non-null float64
Proline                         178 non-null int64
dtypes: float64(11), int64(3)
memory usage: 19.5 KB

print(data['category'].value_counts())
2    71
1    59
3    48
```

### 2.2 案例实验

#### 2.2.1 数据预处理
    
    由于特征都是连续特征，因此这里根据不同类别的红酒，对数据进行分组，查看数据的盒须图

**连续特征分析:**  

![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Alcohol.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Malic%20acid.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Ash.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Alcalinity%20of%20ash.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Magnesium.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Total%20phenols.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Flavanoid.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Nonflavanoid%20phenols.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Proanthocyanins.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Color%20intensity.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Hue.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/OD280OD315%20of%20diluted%20wines.jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/Proline.jpg)

    通过盒须图可以看出，不同类别的红酒的成分还是有较大差别
    
    并且，可以注意到：
    几乎所有特征都一部分的离群点，因此我们需要对异常值进行处理

**异常值处理:**

    异常值出现的原因可能有很多，由于这里不了解红酒的业务和工艺，因此无法进行因素分析
    我们可以进行简单的推测，即酿造时间、酸碱度等等原因。
    
    如果某一个特征出现了利群，可能是正常的，但是如果多种特征都有较大的离群效应
    那么，可能是这个红酒的品质有一定问题，我们需要去除掉这部分的样本

``` python
def find_outlier(data,features,label,n):
    '''
    寻找异常点
    Paramters：
        data:                数据集
        features：           特征
        label：              类标号
        n:                   设置有n个异常的特征时，认为是异常点
    Return：
        multiple_outliers：  异常值的list
    '''
    grouped = data.groupby(data[label])
    outlier_list = []
    for col in features:
        for name,group in grouped:
            Q1 = np.percentile(group[col], 25)
            Q3 = np.percentile(group[col], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            tmp = group[(group[col] < Q1 - outlier_step)|(group[col] > Q3 + outlier_step)].index.tolist()
            outlier_list.extend(tmp)
    outlier_indices = Counter(outlier_list)
    multiple_outliers=list(k for k,v in outlier_indices.items() if v >= n)
    return multiple_outliers
```
    总样本有 178 个
    有离群点的特征的样本有 49 个
    有两个以上离群的特征的样本有 13 个
    有三个以上离群的特征的样本有 5 个
    有四个以上离群的特征的样本有 1 个

    由于总样本较少，因此，我们选择去掉有三个以上离群特征的样本
    去除之后，样本还有173个

#### 2.2.2 特征工程

**特征间的相关性分析:**

    由于朴素贝叶斯的特性，我们要尽量保证特征间无关（朴素）
    首先来看特征间的相关性
    由于 Flavanoid 和 Total phenols 间相关性极高，并且和 Nonflavanoid phenols 呈极高的负相关性
    因此，可以认为存在一定的特征冗余，我们这里选择去除 Flavanoid
    
<img src='https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/corr.jpg' width= 60% height= 60% />

**特征与类标号的相关性分析:**

    同时，我们要保证特征与类标号存在一定的相关性
    再来看特征与类标号之间的相关性
    在通过皮尔逊相关系数，可以看出 Ash 和类标号之间相关性极低，可以说不相关
    因此，这里我们选择去除 Ash 特征

``` python
                         columns              corr_value
6                      Flavanoid    [0.8789611136418785]
11  OD280/OD315 of diluted wines    [0.8020532725664187]
5                  Total phenols    [0.7278447065178943]
12                       Proline    [0.6345172262547121]
10                           Hue    [0.6176699048899034]
3              Alcalinity of ash    [0.5714645341403812]
7           Nonflavanoid phenols    [0.5089280017938433]
8                Proanthocyanins   [0.49805592740689575]
1                     Malic acid   [0.43972272685214164]
0                        Alcohol   [0.32233727967496606]
9                Color intensity   [0.26739993328200146]
4                      Magnesium    [0.2128703266316483]
2                            Ash  [0.031039521734865475]
```
    
**特征散点矩阵图:**

<img src = 'https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/NaiveBayes/matrix.jpg' width= 70% height= 70% />

    可以看出，有一些特征间是线性相关的
    但是整体而言，所有特征都符合高斯分布
    
    可以注意到：
    三种不同类别的样本，虽然有一定交叉重叠的部分，但是边界比较清晰，应该用线性分类器就能分开

**数据标准化:**

    由于整个数据集都是连续特征并呈现正态分布，一般来说应该选择z-score标准化
    但这里，由于数据集较小，并且采用朴素贝叶斯模型，并不是特别需要对数据进行标准化
    因此，省略该步骤

#### 2.2.3 模型训练与评估

``` python

def classification(x,y):
    '''
    利用高斯贝叶斯对数据集进行分类测试
    Paramters：
        x:           输入特征空间
        y:           预测空间
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,random_state=6)
    model = GaussianNB()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    print('accuracy score:', model.score(x_test,y_test))
    print(metrics.classification_report(expected, predicted)) 
    print(confusion_matrix(expected, predicted, sample_weight=None))

classification(x,y)    
accuracy score: 1.0
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        20
           3       1.00      1.00      1.00        13

   micro avg       1.00      1.00      1.00        52
   macro avg       1.00      1.00      1.00        52
weighted avg       1.00      1.00      1.00        52

[[19  0  0]
 [ 0 20  0]
 [ 0  0 13]]
```
    这里直接采用高斯贝叶斯进行分类，也没有太多的参数可调
    由于样本较少、模型比较简单，因此准确率很高，达到了100%

## 3 总结

    贝叶斯是概率论衍生的模型，非常的经典
    学过概率和数理统计的同学，应该都非常喜欢这种模型，理解简单，而且经得起推敲，模型很容易理解解释
    
    这里由于是连续特征，因此用了高斯贝叶斯，除此之外还有伯努利贝叶斯、多项式贝叶斯等等
    贝叶斯信念网络也是贝叶斯模型很重要的一块，后面学习过后再来试一下
    
    在实际的工业界中，朴素贝叶斯也是非常的常用
    它简单，训练时间短，判断快，而且准确率还不错，是很有生命力的模型
