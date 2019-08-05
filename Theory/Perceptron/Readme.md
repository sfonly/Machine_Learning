  * [感知机算法](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机算法)
    * [感知机的原理](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#1-感知机的原理)
      * [模拟神经元](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#11-模拟神经元)
      * [感知机的判别机理](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#12-感知机的判别机理)
    * [距离度量](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#2-距离度量)
      * [点到线的距离](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#21-点到线的距离)
      * [点到面的距离](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#22-点到面的距离)
    * [感知机模型](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#3-感知机模型)
      * [感知机的数学表达](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#31-感知机的数学表达)
      * [感知机的损失函数](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#32-感知机的损失函数)
    * [感知机参数学习方法](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#4-感知机参数学习方法)
      * [原始形式](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#41-原始形式)
      * [对偶形式](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#42-对偶形式)
      * [两种方法的选择](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#43-两种方法的选择)
    * [感知机算法代码实现](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#5-感知机算法代码实现)
    * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#6-总结)

    
# 感知机算法       

  **感知机是由 Rosenblatt 在1957年提出的一种方法，是神经网络和支持向量机的基础。**

## 1 感知机的原理

[维基百科-感知机](https://zh.wikipedia.org/zh-hans/感知器)  

### 1.1 模拟神经元

  **我们知道，感知机其实就是模拟一个单独的神经元的工作方式，得出一个是还是否的判断。**
  
  <img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/感知机模型.png" width = 30% height = 30% />
  
    输入:  X = (1, a1, a2, a3, ... , an) 
          W = (w0, w1, w2, w3, ... , wn)
          
  **当我们把输入转化为向量时，就可以用向量和权重的内积计算出一个数值，再利用sigmoid函数的性质，将其转化为布尔类型的判断，这就是感知机**

### 1.2 感知机的判别机理

  **我们将样本的特征向量在空间中表示出来，如下图二维的特征所示：**
  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点线面.png" width = 50% height = 50% />

  **在上图的问题中，感知机就是那一条线，我们要找的就说能够将问题分开的这这条‘线’**

    感知机是一种线性模型，一般用于二分类，其输入是实例的特征向量，输出的是事例的类别label，分别是 +1 和 -1，属于判别模型的一种。  
    通常而言，我们认为可以通过一条线将二维空间中的点（数据）分为两边，即两个类别
    如果上升到三维、甚至更高维的空间中，能够将其分开的面，即为 “超平面”
    注意：如果是非线性可分的模型，则可能没有超平面！

## 2 距离度量
  
  **那么，如何寻找这个超平面呢？**  
  **大师们的想法是：**  
  
    在空间中，分别计算+1和-1的点到超平面的距离。  
    被这个超平面正确分类的点的距离记为0  
    然后，将所有被这个超平面错分的点的距离相加，记为这个超平面的误差  
    如果，所有的点都被正确分类，那么这个误差就应该为0
    如果，有没有被正确分类的点，那么就让这个误差尽量的小
    
  **这样，就把一个人类理想的模式识别的问题，转化为了几何数学的问题。**   
  **从一个人类直觉的模型，转化为了数学公式，多么天才的想法！**  
  
### 2.1 点到线的距离

公式中的直线方程为 Ax+By+C=0，P点的坐标为(x0,y0)

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点到直线距离公式.png" width = 20% height = 20% />
    
### 2.2 点到面的距离
假设超平面是 h=w⋅x+b，其中 w=(w0,w1,...wm), x=(x0,x1,...xm)，样本点 x′ 到超平面的距离如下：

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点到平面.png" width = 20% height = 20% />

## 3 感知机模型
### 3.1 感知机的数学表达

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点到超平面.png" width = 40% height = 40% />

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/感知机定义公式.png" width = 30% height = 30% />

### 3.2 感知机的损失函数

在感知机中，我们定义：  
    
    1. 对于任意一个样本点（xi,yi）,我们可以通过其距离公式计算器误差
    2. 如果 w*xi+b>0, 则记 yi = +1
    3. 如果 w*xi+b<0, 则记 yi = -1
    4. 所以，我们只需要保证 yi*(w*xi+b)>0 ,则是样本被正确分类
    5. 而被错误分类的情况是： yi*(w*xi+b)<0

**因此，在感知机中，我们可以定义损失函数为：** 

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/损失函数.png" width = 30% height = 30% />

**需要特别注意的是：** 

    M 是当前分类器情况下，被错分的样本域
    因此，只有被错分的样本才会进行参数优化
    而被正确分类的样本对模型优化无影响

## 4 感知机参数学习方法
  
  **感知机学习的过程，既是参数 W 和 b 优化的过程，通过将损失函数极小化，得到最优的参数，从而的出感知机的参数模型。**

**目标函数如下：** 

<img src="https://latex.codecogs.com/gif.latex?\min&space;L(w,b)&space;=&space;\min(-&space;\sum_{x&space;\epsilon&space;M}&space;y_{i}(w\cdot&space;x_{i}&space;&plus;&space;b))" title="\min L(w,b) = \min(- \sum_{x \epsilon M} y_{i}(w\cdot x_{i} + b))" />

    在实际应用中，一般不会采用批量梯度下降算法，因为批量梯度下降算法需要每次带入所有的样本
    而在感知机中，被正确分类的样本不会对参数有影响，只有错分样本才能参与模型优化
    因此，我们通常会采用随机梯度下降，每次随机选择一个样本点进行参数优化，这样可以极大的提升算法运行效率

### 4.1 原始形式

**采用梯度下降法优化参数 w 和 b，分别对目标函数求偏导令其等于 0，可得：**

<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;w&space;&plus;&space;\eta&space;y_{i}&space;x_{i}" title="w = w + \eta y_{i} x_{i}" />  

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;b&space;&plus;&space;\eta&space;y_{i}" title="b = b + \eta y_{i}" />

**在上述情况下，参数最优化，损失函数最小，训练误差最小**

**伪代码如下:**

```python
input:  TrainingData: {(x0,y0),(x1,y1), ...,(xn,yn)} # xi为特征向量， y= +1 or -1
        learning_step:  0<lr<1  # 学习速率
        max_iter:  n # 最大迭代次数
output：w,b
        f(x)=sign(wx+b)
class Train:
  1. 赋初始值 w0, b0
  2. 选择迭代的样本点 (xi, yi)
  3. if yi*(w*xi+b)<0:
       更新 w, b:
          w = w + lr*yi*xi
          b = b + lr*yi
  4. 如果，已经达到最大迭代次数，或者样本中点都被分对，则停止
     否则，回到 2 继续循环
```

### 4.2 对偶形式

原始-对偶问题，是基于凸优化的思想，最小二乘、线性规划等都是凸优化问题  
[维基百科-凸优化](https://zh.wikipedia.org/wiki/凸優化)

    关于对偶，可以简单理解为，将原始问题在某个可行域上，转化为一个等价的对偶问题，从而更加方便的得出答案

**根据 w 和 b 的梯度更新公式有：**

<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;w&space;&plus;&space;\eta&space;y_{i}&space;x_{i}" title="w = w + \eta y_{i} x_{i}" />  

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;b&space;&plus;&space;\eta&space;y_{i}" title="b = b + \eta y_{i}" />

**在 n 次迭代后，被错分的点(xi, yi)属于M数据集，那么 w 和 b 的梯度下降公式应该为：**

<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;w&space;&plus;&space;\sum_{{x_{i}}&space;\in&space;M}&space;{}&space;\eta&space;{x_{i}}&space;{y_{i}}" title="w = w + \sum_{{x_{i}} \in M} {} \eta {x_{i}} {y_{i}}" />

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;b&space;&plus;&space;\sum_{{x_{i}}&space;\in&space;M}&space;{}&space;\eta&space;{y_{i}}" title="b = b + \sum_{{x_{i}} \in M} {} \eta {y_{i}}" />

**如果初始设置的 w 和 b 为 0，总的迭代次数是 n 次，那么，我们可以将 w 和 b 的梯度下降公式进行如下变换，整理成其对偶形式：**

<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;w&space;&plus;&space;\sum_{{x_{i}}&space;\in&space;M}&space;{}&space;\eta&space;y_{i}&space;x_{i}&space;=&space;\sum_{i=1}^{n}&space;[(n_{i}\eta&space;)y_{i}&space;x_{i}]" title="w = w + \sum_{{x_{i}} \in M} {} \eta y_{i} x_{i} = \sum_{i=1}^{n} [(n_{i}\eta )y_{i} x_{i}]" />

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;b&space;&plus;&space;\sum_{{x_{i}}&space;\in&space;M}&space;{}&space;\eta&space;y_{i}&space;=&space;\sum_{i=1}^{n}&space;[(n_{i}\eta&space;)y_{i}]" title="b = b + \sum_{{x_{i}} \in M} {} \eta y_{i} = \sum_{i=1}^{n} [(n_{i}\eta )y_{i}]" />

**我们注意到： ni 实际上是参数当前修改的迭代次数，那么我们令：**

<img src="https://latex.codecogs.com/gif.latex?\alpha&space;=&space;n_{i}&space;\eta" title="\alpha = n_{i} \eta" />

**那么，就可以将梯度下降公式转化为：**

<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;\sum_{i=1}^{n}&space;[(n_{i}\eta&space;)y_{i}&space;x_{i}]&space;=&space;\sum_{i=1}^{n}&space;\alpha_{i}&space;y_{i}&space;x_{i}" title="w = \sum_{i=1}^{n} [(n_{i}\eta )y_{i} x_{i}] = \sum_{i=1}^{n} \alpha_{i} y_{i} x_{i}" />

<img src="https://latex.codecogs.com/gif.latex?b&space;=&space;\sum_{i=1}^{n}&space;(n_{i}\eta&space;)y_{i}&space;=&space;\sum_{i=1}^{n}&space;\alpha_{i}&space;y_{i}" title="b = \sum_{i=1}^{n} (n_{i}\eta )y_{i} = \sum_{i=1}^{n} \alpha_{i} y_{i}" />

**这样，我们就得出了感知机的对偶算法**

**感知机对偶算法的伪代码如下：**

```python
input:  TrainingData: {(x0,y0),(x1,y1), ...,(xn,yn)} # xi为特征向量， y= +1 or -1
        learning_step:  0<lr<1  # 学习速率
        max_iter:  n # 最大迭代次数
output：α,b
        w= sum(αj*yj*xj)*x ; b = sum(αj*y) ;
        f(x) = sign(w*x + b) = sign(sum(αj*yj*xj)*x + b)
        # 其中α=(α1,α2,...,αn)T
class Train:
  1. 赋初始值 α0, b0
  2. 选择迭代的样本点 (xi, yi)
  3. if yi*(sum(αj*yj*xj)*xi + b)<0:
       更新 α, b:
          αi = αi + lr
          b = b + lr*yi
       # 注意j是更新迭代的次数，i是指样本中第i个点
  4. 如果，已经达到最大迭代次数，或者样本中点都被分对，则停止
     否则，回到 2 继续循环
```

**注意代码，实际上样本训练集的特征，只在判断条件中使用到，其余都没有使用到！**  
**我们将判断条件提取出来：**

<img src="https://latex.codecogs.com/gif.latex?y_{i}((\sum_{j=1}^{n}\alpha&space;_{j}&space;y_{j}&space;x_{j})\cdot&space;x_{i}&space;&plus;&space;b)&space;\leq&space;0&space;\Leftrightarrow&space;y_{i}(\sum_{j=1}^{n}\alpha&space;_{j}&space;y_{j}(x_{j}\cdot&space;x_{i})&space;&plus;&space;b)&space;\leq&space;0" title="y_{i}((\sum_{j=1}^{n}\alpha _{j} y_{j} x_{j})\cdot x_{i} + b) \leq 0 \Leftrightarrow y_{i}(\sum_{j=1}^{n}\alpha _{j} y_{j}(x_{j}\cdot x_{i}) + b) \leq 0" />

**我们每次做运算的时候，只需要先计算好所有的 xj 和 xi 的内积，形成一个矩阵，就是 Gram 矩阵**  
**当我们再次使用进行判断时，只需要根据对应的 i 和 j 查询即可**  

<img src="https://latex.codecogs.com/gif.latex?Gram&space;=&space;[x_{i}&space;\cdot&space;x_{j}]_{N\times&space;N}" title="Gram = [x_{i} \cdot x_{j}]_{N\times N}" />

### 4.3 两种方法的选择

    那么，我们什么时候用原始形式，什么时候利用对偶形式呢?
    可以注意到，区别就在于判断条件！

**原始形式：**

<img src="https://latex.codecogs.com/gif.latex?y_{i}(w^{T}&space;\cdot&space;x_{i}&space;&plus;&space;b)&space;\leq&space;0" title="y_{i}(w^{T} \cdot x_{i} + b) \leq 0" />

**对偶形式：**

<img src="https://latex.codecogs.com/gif.latex?y_{i}(\sum_{j=1}^{n}\alpha&space;_{j}&space;y_{j}(x_{j}\cdot&space;x_{i})&space;&plus;&space;b)&space;\leq&space;0" title="y_{i}(\sum_{j=1}^{n}\alpha _{j} y_{j}(x_{j}\cdot x_{i}) + b) \leq 0" />

**时间复杂度：**  
    
    原始形式：
        需要计算权重 w 和样本 x 的内积，时间复杂度为：O(n)，n 为特征向量数量
    对偶形式：
        可以提前计算 Gram 矩阵，计算 Gram 矩阵的时间复杂度为：O(N), N为样本数量
        而在训练模型的时候，则只需要查询即可，时间复杂度为 O(1)

**结论：**

    1. 如果特征向量n过大，那么计算 (w,x) 的内积的时间会过长。这时候采用对偶形式 
    2. 如果样本数量N巨大，那么计算 Gram 矩阵时间很长，所占据的内存空间也极大，这时候就应该采用原始形式。

**对偶形式的感知机算法，把每轮迭代的时间复杂度从特征空间维度 n 转移到了样本数据数量 N 上，这真的是一件很奇妙的做法。**

## 5 感知机算法代码实现

    我自己用python实现了一遍感知机算法，仅供参考：
    
[感知机算法python实现](https://github.com/sfonly/Machine_Learning/tree/master/Algorithm/Perceptron)

## 6 总结

    花了很长的心思来写感知机的理论学习笔记。
    感知机虽然简单，但是大名鼎鼎的 SVM 和 神经网络 都是从它身上衍伸出来的
    重点不在于算法本身的应用，而是它背后的数学思想，值得好好的学习。
    
引用：    
李航老师 《统计学习方法》  
[刘建平-感知机原理小结](https://www.cnblogs.com/pinard/p/6042320.html)  




