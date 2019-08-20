* [朴素贝叶斯案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes)
  * [NaiveBayes原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#1-NaiveBayes原理)
  * [wine案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#2-wine案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#22-案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#221-数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#222-特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#223-模型训练与评估)
      * [结果可视化](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#224-结果可视化)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/NaiveBayes#3-总结)


# 朴素贝叶斯案例学习笔记
## 1 NaiveBayes原理
**NaiveBayes定义：**  
[支持向量机-维基百科](https://zh.wikipedia.org/wiki/支持向量机)



**优点：**  


**缺点：**  



## 2 Wine案例
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
    


#### 2.2.2 特征工程

**特征相关性分析:**


**连续特征分析:**


**数据标准化:**



#### 2.2.3 模型训练与评估



#### 2.2.4 结果可视化 




## 3 总结

    
