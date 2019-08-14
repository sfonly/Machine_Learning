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



### 2.2 案例实验

#### 2.2.1 数据预处理
    
**去除空值:**



#### 2.2.2 特征工程

**特征相关性分析:**


**去除异常值:**


**数据归一化:**


#### 2.2.3 模型训练与评估


#### 2.2.4 结果可视化 
    

## 3 总结

