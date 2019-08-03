  * [感知机算法](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#感知机算法)
    * [感知机的原理](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#感知机的原理)
    * [距离度量](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#距离度量)
      * [点到线的距离](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#点到线的距离)
      * [点到面的距离](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#点到面的距离)
    * [感知机模型](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#感知机模型)
    * [感知机学习算法](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#感知机学习算法)
    * [总结](https://github.com/sfonly/Machine_Learning/new/master/Theory/Perceptron/Readme.md/#总结)
    
# 感知机算法       
    感知机是由 Rosenblatt 在1957年提出的一种方法，是神经网络和支持向量机的基础。
## 感知机的原理
[维基百科-感知机](https://zh.wikipedia.org/zh-hans/感知器)  

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
    
    
### 点到面的距离
假设超平面是 h=w⋅x+b，其中 w=(w0,w1,...wm), x=(x0,x1,...xm)，样本点 x′ 到超平面的距离如下：

## 感知机模型


## 感知机学习算法


## 总结


















