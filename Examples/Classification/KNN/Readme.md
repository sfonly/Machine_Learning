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

    KNN 就是通过计算空间中向量的欧式距离，根据其最近的 K 个向量的类别，预测其类别的一种方法

**优点：**  
1. 模型简单，可以用于分类和回归  
2. 可用于数值数据，也可用于离散数据和标称属性数据
3. 随着数据趋于无限，算法保证错误率不会超过贝叶斯算法错误率的两倍
4. 对异常值不敏感
5. KNN能够形成稳定的决策边界（可通过 1NN 的特例来观察）
6. 理论和实践均非常成熟，工程实践中广泛应用

**缺点：**  
1. 数据量大时，计算时间过长，而且需要极大的内存
2. 是非参数学习法，每次使用时都需要重新训练模型
3. 样本不平衡的时候，可能有较大误差
4. 无法给出同一类别数据的内在联系

## 2 Hellen案例
### 2.1 案例背景

    Hellen 经常使用约会网站寻找猎物用于解决自己的终身大事。（PS：女猎人啊！！！）
    
    时光流逝，Hellen还是单身狗。
    她遇见了很多约会对象，有好有坏，Hellen 将约会网站上的人分为三类：不喜欢的人，魅力一般的人和极具魅力的人。
    为了以后的约会能尽可能与极具魅力的人在一起，Hellen需要通过一些指标对约会对象进行分类。

    Hellen收集了一堆数据，每个样本包含三个特征：每年的飞行常客里程数，玩视频游戏所耗时间百分比和每周消费的冰淇淋公升数。  

**数据集描述:**

|      |feature_name      | feature_type | structure | describe           |
| ---- | :----:           | :----:       | :----:    | :----:             |
| 0 | flight_mileage      | continues    | float     | 每年的航空里程数     |
| 1 | games_time_percent  | continues    | float     | 每天玩游戏的时间比率 |
| 2 | eat_icecream_liters | continues    | float     | 每周吃冰淇淋的公升数 |
| 3 | label               | discrete     | norminal  | hellen是否喜欢      |


**预测值描述:**

``` python
print(ORGdata.label.unique())
print(ORGdata[ORGdata['label'] == 'largeDoses'].shape[0])
print(ORGdata[ORGdata['label'] == 'smallDoses'].shape[0])
print(ORGdata[ORGdata['label'] == 'didntLike'].shape[0])

['largeDoses' 'smallDoses' 'didntLike']
327
331
342
```

### 2.2 案例实验

#### 2.2.1 数据预处理
    
**去除空值:**

``` python
print(data.isnull().sum()) # 通过分析可以看出，数据集无缺失值
flight_mileage         0
games_time_percent     0
eat_icecream_liters    0
label                  0
dtype: int64
```

#### 2.2.2 特征工程

**特征相关性分析:**

    Hellen 不同等级的约会对象，与航空里程数的盒须图   

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/flight_mileage.jpg" width=40% height=40% />

    Hellen 不同等级的约会对象，与每天玩游戏时间占比的盒须图   
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/games_time_percent.jpg" width=40% height=40% />

    Hellen 不同等级的约会对象，与每周吃冰淇淋的公升数的盒须图  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/eat_icecream_liters.jpg" width=40% height=40% />

**去除异常值:**

    通过前文的盒须图分析，极少部分样本点是离群的，且这部分样本点只是单独某个属性离群，因此不认为是异常值

**数据归一化:**

    这里我自己实现了归一化方法，没有做数据校验，只能处理数值数据
    
#### 2.2.3 模型训练与评估
``` python
# 进行训练
k = 8
predicted = predict(train_feature, test_feature, train_labels, k)
tp,pre_labels = classify(train_feature, test_feature, train_labels, test_labels, k)
print('KNN predict:', predicted)
print('KNN accuracy:',tp)
```
    尝试了几次，最后发现 K=8 的时候，准确率最高，且速度还较快
    KNN accuracy: 0.9567
    不想重构classify方法了

#### 2.2.4 结果可视化 
    
    flight_mileage-games_time_percent散点图  
    x轴为flight_mileage，y轴为games_time_percent  
    不同的颜色为真实的类别标签  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-test01.jpg" width=50% height=50% />

    flight_mileage-games_time_percent散点图  
    x轴为flight_mileage，y轴为games_time_percent  
    不同的颜色为预测的类别标签  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-pre01.jpg" width=50% height=50% />

    games_time_percent-eat_icecream_liters散点图  
    x轴为games_time_percent，y轴为eat_icecream_liters
    不同的颜色为真实的类别标签  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-test12.jpg" width=50% height=50% />

    games_time_percent-eat_icecream_liters散点图  
    x轴为games_time_percent，y轴为eat_icecream_liters
    不同的颜色为预测的类别标签  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Examaples/Classification/KNN/test-pre12.jpg" width=50% height=50% />

## 3 总结

    KNN真的很简单
    需要格外注意，KNN的本质是距离相似度
    需要根据应用数据的情况选择合适的相似度计算方法：欧式距离、巴氏距离、曼哈顿距离、切比雪夫距离等等
    针对KNN的计算量的问题，有很多的优化方法，主要是通过建设不必要的距离评价来实现的
