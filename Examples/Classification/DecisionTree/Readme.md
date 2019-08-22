* [决策树案例学习笔记](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/DecisionTree#决策树案例学习笔记)
  * [决策树原理](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/DecisionTree#1-决策树原理)
  * [Iris案例](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/DecisionTree#2-Iris案例)
    * [案例背景](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/DecisionTree#21-案例背景)
    * [案例实验](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/DecisionTree#22-案例实验)
  * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Examples/Classification/DecisionTree#3-总结)


# 决策树案例学习笔记
## 1 决策树原理

[决策树-维基百科](https://zh.wikipedia.org/wiki/决策树)

**决策树定义：**  

    决策树是一种树形结构。
    其中，每个内部节点表示一个属性上的特征情况，每个分支代表该节点特征的不同输出，而每个叶子节点代表一种类别。
    决策树算法就是要利用已有信息，来构造一个决策树。
    对于测试信息的分类，将按照决策上的各个分支来进行，满足对应条件，就将进入到该分支，直到叶子节点。

**如何选取分支条件，构造一颗高效的决策树呢?**   

    一般算法中都是以信息熵为判定标准。
    信息论之父香农借鉴了热力学的概念，把信息中排除了冗余后的平均信息量称为“信息熵”，并给出了计算信息熵的数学表达式。  

<img src = 'https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563877701(1).jpg' width = 40% height=40% />

**信息熵即是表示事件所包含信息量大小的概念。  
我们所知事件的信息量越小，事件的发生概率越不确定，信息熵越大；  
我们所知事件的信息量越多，事件发生的概率越确定，信息熵越小**    

    1. 太阳从东方升起  
     这是一个确定事件，一定会发生，概率为1，通过这句话，我们几乎不能获得新的信息。
     因此，我们认为信息量很小，信息熵为0。  
     
    2. 中国足球队将参加2022年世界杯并取得冠军  
     几乎是一个确定事件，一定不会发生，概率约等于0，通过这句话，我们几乎不能获得新的信息，我们认为信息量很小，信息熵为0。  
     
    3. 一群好朋友，天气好，就去打篮球，天气差，就不去。这个情况下，我们知道了明天的天气是否要下雨。  
     原来的事件是一个混沌的情况，即不知道天气的情况下，不知道这个出去打篮球这个事件会不会发生，概率为50%，信息熵为1。
     当我们知道了明天要下雨，那么打篮球这个事件就不会发生了，概率为0%，信息熵也变为了0，不下雨的情况同理；
     通过获得明天天气的这个信息，将原来不确定的系统，变为了一个确定的系统。
     降低了混乱程度，使得信息的熵值变为了0，获得的信息增益为1，即这个事件的带来的信息量极大，信息增益极大。

        Gain = Entroy（旧系统）- Entroy（新系统）=1

**在决策树算法中，找出对预测事件信息增益大的属性来构造分支，通过这样的方式就能选择合适的建立分枝的属性，从而构造一个高效的决策树**  

**优缺点：**  

    三种决策树中，ID3和C4.5是用信息增益进行计算，并且都是采用局部最优化算法来选择分枝属性。  
    
    1. ID3算法
        ID3不能处理连续数据，需要把连续数据进行分箱离散化后才能处理。
        ID3会在前几层分枝过程中，会优先选择分枝数多的，能分得更完全的特征来建树。
        这会导致树的宽度极大，在实际的树模型实现中，检索效率较低，且分类效果不佳。
    
    2. C4.5算法
        C4.5算法在ID3的基础上做了一定的改进，可以处理连续数据
        采用了信息增益率来解决选择分枝特征值过多的问题
        在缺失值和异常值上变现较好
    
    3. CART算法
        CART是分类回归树，可用于解决分类问题和回归问题
        也是随机森林中最常用的树型结构
        CART一定是一个二叉树
        可以自动寻找连续数据最优切割点
        并且可以重复利用某个特征（切分得更细）

**需要注意的是：采用单一的决策树模型，需要谨慎的进行剪枝来处理过拟合问题**

## 2 Iris案例
### 2.1 案例背景

    鸢尾花数据集可能是模式识别、机器学习等领域里被使用最多的一个数据集了，很多教材用这份数据来做案例。
    鸢尾花数据集最初由Edgar Anderson 测量得到
    而后在著名的统计学家和生物学家R.A Fisher的文章「The use of multiple measurements in taxonomic problems」中被使用

    数据中的两类鸢尾花记录结果是在加拿大加斯帕半岛上
    于同一天的同一个时间段，使用相同的测量仪器，在相同的牧场上由同一个人测量出来的。
    这是一份有着70年历史的数据，虽然老，但是却很经典。  

**特征描述:**

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class: 
    * Iris Setosa
    * Iris Versicolour
    * Iris Virginica

**特征统计分析:**

    data.describe()

|       |sepal_length | sepal_width | petal_length | petal_width|
| ----  | :----:      | :----:      | :----:       | :----:     |
|count  |  150.000000 |  150.000000 |   150.000000 |  150.000000|
|mean   |    5.843333 |    3.054000 |     3.758667 |    1.198667|
|std    |    0.828066 |    0.433594 |     1.764420 |    0.763161|
|min    |    4.300000 |    2.000000 |     1.000000 |    0.100000|
|25%    |    5.100000 |    2.800000 |     1.600000 |    0.300000|
|50%    |    5.800000 |    3.000000 |     4.350000 |    1.300000|
|75%    |    6.400000 |    3.300000 |     5.100000 |    1.800000|
|max    |    7.900000 |    4.400000 |     6.900000 |    2.500000|

**类标号分析:**
分为三类，每个类别都是50个

### 2.2 案例实验

**数据预处理:**  
    
    data.isnull().sum() # 通过分析可以看出，数据集无缺失值
    
**特征工程:**  

    查看各特征的盒须图：
    
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563877428(1).jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563877462(1).jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563877471(1).jpg)
![loss](https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563877485(1).jpg)

    可以发现部分特征有超出上下线的异常，但也可能鸢尾花的性状就是这样，比如花骨朵小一点什么的
    我认为只有同时具备了两个以上的异常特征时，才算作是异常点，予以去除。
    经分析，不存在这样的异常点
    
    然后利用散点柱状矩阵，查看三种鸢尾花的特征关联性
    
<img src = 'https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563877498(1).jpg' width = 50% height=50% />

    可以看出：
    Iris-setosa 这种鸢尾花和另外两种鸢尾花差别较大
    另外两种鸢尾花虽然有一定重叠，但是边界较明显，属于线性可分的范畴（因此，鸢尾花数据集在线性分类器上表现也较好）
    由于数据集和特征较少，这里就不进行特征关联性分析和特征降维
    
**模型训练与评估:**  

```python
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
        max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, presort=False, random_state=None,
        splitter='best')
print('clf.score:',clf.score(X_test,y_test))
clf.score: 0.9777777777777777
```

    利用sklearn里model_selection里的函数切割数据集，70%作为训练集，30%作为测试集
    在sklearn里，DecisionTree默认使用gini系数，我这里使用了信息增益
    由于数据集较小，模型比较简单，基本上没怎么调参，也不需要进行剪枝
    最后模型的的准确率为0.977左右，由于有随机数，每次的分类准确率有一定浮动

**模型可视化:**  
    
    决策树属于易于理解，符合人类思维逻辑的算法之一，可视化效果也较好
    这里我通过graphviz包实现了决策树图的可视化，需要在本地电脑安装graphviz包，并将其启动路径加入到系统环境变量中

<img src = 'https://github.com/sfonly/Machine_Learning/blob/master/img_folder/examples/1563879621(1).jpg' width = 80% height=80% />

## 3 总结

    我非常的喜欢决策树，符合人的思维模型。
    决策树是数据挖掘、机器学习基础算法之一，广泛应用到诸多领域当中（几乎所有领域）。

    理解建树的流程，如何选择特征？如何找到连续特征的最佳切割点？如何防止过拟合？
    这些是学习决策树的重点。
    然后实现决策树的优化算法和递归实现方式也是很重要的。

    决策树虽然简单，但是很常用，并RandomForest集成算法中表现极好。
    而且，C4.5和CART两个树算法都是数据挖掘十大算法之一，可见其重要性
