  * [感知机算法](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机算法)
    * [感知机的原理](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机的原理)
    * [距离度量](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#距离度量)
      * [点到线的距离](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#点到线的距离)
      * [点到面的距离](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#点到面的距离)
    * [感知机模型](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机模型)
      * [感知机的数学表达](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机的数学表达)
      * [感知机的损失函数](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机的损失函数)
    * [感知机参数学习方法](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#感知机参数学习方法)
      * [原始形式](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#原始形式)
      * [对偶形式](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#对偶形式)
    * [总结](https://github.com/sfonly/Machine_Learning/tree/master/Theory/Perceptron#总结)
    
    
# 感知机算法       
    感知机是由 Rosenblatt 在1957年提出的一种方法，是神经网络和支持向量机的基础。
## 感知机的原理
[维基百科-感知机](https://zh.wikipedia.org/zh-hans/感知器)  

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点线面.png" width = 50% height = 50% />

    感知机是一种线性模型，一般用于二分类，其输入是实例的特征向量，输出的是事例的类别label，分别是 +1 和 -1，属于判别模型的一种。  
    通常而言，我们认为可以通过一条线将二维空间中的点（数据）分为两边，即两个类别
    如果上升到三维、甚至更高维的空间中，能够将其分开的面，即为 “超平面”
    如果是非线性可分的模型，则可能没有超平面

## 距离度量
  **如何寻找这个超平面呢？**  
  **大师们的想法是：**  
  
    在空间中，分别计算+1和-1的点到超平面的距离。  
    被这个超平面正确分类的点的距离记为0  
    然后，将所有被这个超平面错分的点的距离相加，记为这个超平面的误差  
    如果，所有的点都被正确分类，那么这个误差就应该为0
    如果，有没有被正确分类的点，那么就让这个误差尽量的小
    
  **这样，就把一个人类理想的模式识别的问题，转化为了几何数学的问题。**   
  **从一个人类直觉的模型，转化为了数学公式，多么天才的想法！**  
### 点到线的距离
公式中的直线方程为 Ax+By+C=0，P点的坐标为(x0,y0)

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点到直线距离公式.png" width = 20% height = 20% />

    
### 点到面的距离
假设超平面是 h=w⋅x+b，其中 w=(w0,w1,...wm), x=(x0,x1,...xm)，样本点 x′ 到超平面的距离如下：

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点到平面.png" width = 20% height = 20% />


## 感知机模型
### 感知机的数学表达

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/点到超平面.png" width = 40% height = 40% />

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/感知机定义公式.png" width = 30% height = 30% />


### 感知机的损失函数

  在感知机中，我们定义：
  1. 对于任意一个样本点（xi,yi）,我们可以通过其距离公式计算器误差
  2. 如果 W*xi+b>0, 则记 yi = +1
  3. 如果 W*xi+b<0, 则记 yi = -1
  4. 所以，我们只需要保证 yi*(W*xi+b)>0 ,则是样本被正确分类
  5. 而被错误分类的情况是： yi*(W*xi+b)<0

**因此，在感知机中，我们可以定义损失函数为：** 

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/损失函数.png" width = 30% height = 30% />


## 感知机参数学习方法
  感知机学习的过程，既是参数 W 和 b 优化的过程，通过将损失函数极小化，得到最优的参数，从而的出感知机的参数模型。

**目标函数如下：** 

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/损失函数.png" width = 30% height = 30% />


  在实际应用中，一般不会采用批量梯度下降算法，因为批量梯度下降算法需要每次带入所有的样本
  而在感知机中，被正确分类的样本不会对参数有影响，只有错分样本才能参与模型优化
  因此，我们通常会采用随机梯度下降，每次随机选择一个样本点进行参数优化，这样可以极大的提升算法运行效率

### 原始形式
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
  4. 如果，达到迭代次数，或者样本中点都被分对，停止
     否则，回到 2 继续循环
```

### 对偶形式

<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/Perceptron/对偶算法.png" width = 60% height = 60% />

未完待续：

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
          w = w + a*yi*xi
          b = b + a*yi
  4. 如果，达到迭代次数，或者样本中点都被分对，停止
     否则，回到 2 继续循环
```


## 总结


















