* [线性回归案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#线性回归案例学习笔记)
  * [线性回归原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#1线性回归原理)
  * [2HousePrice案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#2HousePrice案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#21案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#22案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#221数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#222特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#223模型训练与评估)
      * [结果可视化](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#224结果可视化)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Regression/LinearRegression#3总结)


# 线性回归案例学习笔记
## 1 线性回归原理
**线性回归定义：**  
[线性回归-维基百科](https://zh.wikipedia.org/wiki/线性回归)

在统计学中，线性回归（linear regression）是利用称为线性回归方程的最小二乘函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。
这种函数是一个或多个称为回归系数的模型参数的线性组合
只有一个自变量的情况称为简单回归，大于一个自变量情况的叫做多元回归

**优点：**  
1. 建模速度快，不需要很复杂的计算，在数据量大的情况下依然运行速度很快  
2. 模型可解释性高，可以根据系数给出每个变量的理解和解释  
3. 模型预测的是连续值
4. 应用广泛，几乎是在所有领域中都需要用到

**缺点：**  
1. 对异常值极为敏感
2. 无法解决非线性问题

## 2 HousePrice 案例
### 2.1 案例背景

波士顿（Boston）是美国马萨诸塞州的首府和最大城市，也是美国东北部的新英格兰地区的最大城市。  
波士顿位于美国东北部大西洋沿岸，创建于1630年，是美国最古老、最有文化价值的城市之一。  
同时，波士顿也是美国房价最高的城市之一。  

假如，你作为一个房屋中介，需要根据房屋的状况设置二手房建议售价。
根据以往的经验，美国房屋的售价和多种因素相关，包括面积、地理位置、环境、房屋建立年限、质量等等。  

  **那么，房屋售价到底和哪些因素相关？相关性有多大，如何衡量？如何准确的给出一个房屋的建议售价？**  

  **房屋中介的述求：**  
  **能不能根据房屋中介前期调研获取的数据，拟合出一个数据模型，输入数据参数后，预测房屋的售价**  


**数据集描述:**

|      |feature_name| feature_type | structure | describe                     |
| ---- | :----:     | :----:       | :----:    | :----:                       |
| 0 | CRIM          | continues    | float     | 区域的平均犯罪率               |
| 1 | ZN            | continues    | float     | 住宅用地超过25000平方英尺的比例 |
| 2 | INDUS         | continues    | float     | 城镇非零售商用土地的比例        |
| 3 | CHAS          | discrete     | int       | 是否接近查理斯河               |
| 4 | NOX           | continues    | float     | 一氧化氮浓度                   |
| 5 | RM            | continues    | float     | 住宅平均房间数                 |
| 6 | AGE_1940      | continues    | float     | 年之前建成的自用房屋比例        |
| 7 | DIS           | continues    | float     | 到波士顿五个中心区域的加权距离   |
| 8 | RAD           | continues    | int       | 辐射性公路的接近指数            |
| 9 | TAX           | continues    | int       | 每10000美元的全值财产税率       |
|10 | PTRATIO       | continues    | float     | 区域师生比                     |
|11 | B             | continues    | float     | 每千人中黑人的占比的相关变量     |
|12 | LSTAT         | continues    | float     | 人口中地位低下者的比例          |
|13 | MEDV          | continues    | float     | 自住房的平均房价                |

**预测值描述:**

MEDV： 自住房的平均房价  

    data.MEDV.describe()
    
    count    506.000000
    mean      22.532806
    std        9.197104
    min        5.000000
    25%       17.025000
    50%       21.200000
    75%       25.000000
    max       50.000000


### 2.2 案例实验

#### 2.2.1 数据预处理
    
**去除空值:**

    print(data.isnull().sum()) # 通过分析可以看出，数据集无缺失值
    
    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    MEDV       0

#### 2.2.2 特征工程

**自变量-因变量关联性分析:**

    DIS 和 MEDV


    LSTAT 和 MEDV


    RM 和 MEDV



**去除异常值:**

``` python
i_ = y[y.MEDV == 50].index.tolist()
dropdata(features,y,i_)
```

**数据归一化/标准化:**

    这里的连续特征的分布较大，需要对其进行归一化或标准化处理。
    由于数据集没有表现出明显的高斯分布或均匀分布的情况，这里我对两种方法都进行了尝试。
    

#### 2.2.3 模型训练与评估

    这里使用了线性回归对模型进行训练

#### 2.2.4 结果可视化 
    
    未完待续...


## 3 总结

