* [KNN案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#KNN)
  * [KNN原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#1-KNN原理)
  * [Hellen案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#2-Hellen案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#22-案例实验)
      * [数据预处理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#221-数据预处理)
      * [特征工程](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#222-特征工程)
      * [模型训练与评估](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#223-模型训练与评估)
      * [结果可视化](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#224-结果可视化)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/KNN#3-总结)


# KNN案例学习笔记
## 1 KNN原理
**KNN定义：**  
[最近邻居法-维基百科](https://zh.wikipedia.org/wiki/最近邻居法)

施工中，未完待续

**优点：**  
1. 模型简单  


**缺点：**  


## 2 Hellen案例
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

    print(data.isnull().sum()) # 通过分析可以看出，数据集无缺失值
    


#### 2.2.2 特征工程

**特征相关性分析:**

    Hellen 不同等级的约会对象，与航空里程数的盒须图  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/flight_mileage.jpg)

    Hellen 不同等级的约会对象，与每天玩游戏时间占比的盒须图  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/games_time_percent.jpg)

    Hellen 不同等级的约会对象，与每周吃冰淇淋的公升数的盒须图  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/eat_icecream_liters.jpg)


**去除异常值:**



**数据归一化:**


    
#### 2.2.3 模型训练与评估


    


    


#### 2.2.4 结果可视化 
    
    flight_mileage-games_time_percent散点图  
    x轴为flight_mileage，y轴为games_time_percent  
    不同的颜色为真实的类别标签  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-test01.jpg)

    flight_mileage-games_time_percent散点图  
    x轴为flight_mileage，y轴为games_time_percent  
    不同的颜色为预测的类别标签  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-pre01.jpg)

    games_time_percent-eat_icecream_liters散点图  
    x轴为games_time_percent，y轴为eat_icecream_liters
    不同的颜色为真实的类别标签  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-test12.jpg)

    games_time_percent-eat_icecream_liters散点图  
    x轴为games_time_percent，y轴为eat_icecream_liters
    不同的颜色为预测的类别标签  
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-pre12.jpg)

## 3 总结

 
