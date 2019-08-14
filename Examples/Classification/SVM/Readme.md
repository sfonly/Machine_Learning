* [SVM案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM)
  * [SVM原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#1-SVM原理)
  * [Heartdisease案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/SVM#2-Heartdisease案例)
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
[最近邻居法-维基百科](https://zh.wikipedia.org/wiki/最近邻居法)


**优点：**  


**缺点：**  


## 2 Heartdisease案例
### 2.1 案例背景



**数据集描述:**

|      |feature_name      | feature_type | structure | describe           |
| ---- | :----:           | :----:       | :----:    | :----:             |
| 0 | flight_mileage      | continues    | float     | 每年的航空里程数     |
| 1 | games_time_percent  | continues    | float     | 每天玩游戏的时间比率 |
| 2 | eat_icecream_liters | continues    | float     | 每周吃冰淇淋的公升数 |
| 3 | label               | discrete     | norminal  | hellen是否喜欢      |


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

