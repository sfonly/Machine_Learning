  * [决策树基础知识](https://github.com/sfonly/Machine_Learning/new/master#决策树基础知识)
    * [什么是决策树](https://github.com/sfonly/Machine_Learning/new/master#什么是决策树)
    * [如何建立决策树](https://github.com/sfonly/Machine_Learning/new/master#如何建立决策树)
    * [信息熵](https://github.com/sfonly/Machine_Learning/new/master#信息熵)
      * [什么是信息熵](https://github.com/sfonly/Machine_Learning/new/master#什么是信息熵)
        * [信息论之父——香农](https://github.com/sfonly/Machine_Learning/new/master#信息论之父香农)
        * [信息量的定义_我的理解](https://github.com/sfonly/Machine_Learning/new/master#信息量的定义_我的理解)
        * [信息熵公式推导](https://github.com/sfonly/Machine_Learning/new/master#信息熵公式推导)
      * [信息熵和信息增益有什么用](https://github.com/sfonly/Machine_Learning/new/master#信息熵和信息增益有什么用)
      * [条件熵](https://github.com/sfonly/Machine_Learning/new/master#条件熵)
      * [信息熵使用示例](https://github.com/sfonly/Machine_Learning/new/master#信息熵使用示例)
    * [常用决策树算法](https://github.com/sfonly/Machine_Learning/new/master#常用决策树算法)
      * [ID3](https://github.com/sfonly/Machine_Learning/new/master#ID3)
      * [C4.5](https://github.com/sfonly/Machine_Learning/new/master#C45)
      * [CART](https://github.com/sfonly/Machine_Learning/new/master#CART)
    * [过拟合与剪枝](https://github.com/sfonly/Machine_Learning/new/master#过拟合与剪枝)  
    * [决策树的其他知识](https://github.com/sfonly/Machine_Learning/new/master#决策树的其他知识)
      * [分类树](https://github.com/sfonly/Machine_Learning/new/master#分类树)
      * [回归树](https://github.com/sfonly/Machine_Learning/new/master#回归树)
      * [集成模型](https://github.com/sfonly/Machine_Learning/new/master#决策树集成模型)


# 决策树基础知识
## 什么是决策树

**决策树相关概念：**    
[维基百科-决策树](https://zh.wikipedia.org/wiki/决策树)

**假设，我们现在有客户购买电脑的数据，能够从下面的数据中推断出，哪些客户是想要购买电脑的吗？**   
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/数据1.jpg" width = 50% height = 50% />

**IF-THEN规则：**  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/IF-Then规则.jpg" width = 80% height = 80% />

**好了，这就是我们的大脑凭着 直觉 建立好的决策树：**  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/决策树1.jpg" width = 50% height = 50% />

## 如何建立决策树

**那么，应该怎么建立决策树呢？**  
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/如何建树.jpg" width = 80% height = 80% />

**思考：**  
    为什么是最小决策树？ ———— 奥卡姆剃刀  
    如何选择属性？如何找到连续属性的最优分割点？————信息熵  

## 信息熵
### 什么是信息熵

**信息熵相关概念：**    
[维基百科-信息熵](https://zh.wikipedia.org/wiki/熵_(信息论))

#### 信息论之父——香农
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/信息熵1.jpg" width = 80% height = 80% />

#### 信息量的定义_我的理解
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/信息熵2.jpg" width = 80% height = 80% />

#### 信息熵公式推导
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/信息熵3.jpg" width = 80% height = 80% />

### 信息熵和信息增益有什么用
    信息增益的意义：
    通过信息熵公式，我们可以衡量事件包含的信息量
    
    对于决策树来说，每个和分类结果有关联的属性，都相当于增加了分类结果的确定性，即我们可以计算每个分枝代表的信息熵：
        加入属性 A 之前，系统的信息熵为： 𝐸𝑛𝑡𝑟𝑜𝑝𝑦  
        加入属性 A 之后，系统的信息熵为： 𝐸𝑛𝑡𝑟𝑜𝑝𝑦_𝐴
    
    系统的信息增益为： 𝐼𝑛𝑓𝑜𝑟𝑚𝑎𝑡𝑖𝑜𝑛_𝐺𝑎𝑖𝑛 = 𝐺𝑎𝑖𝑛(𝐴) = 𝐸𝑛𝑡𝑟𝑜𝑝𝑦 − 𝐸𝑛𝑡𝑟𝑜𝑝𝑦(𝐴)
    
    我们现在有了信息增益之后，要建立决策树，需要做的就是：
    1. 在每个决策环节，计算所有属性对系统的信息增益
    2. 然后，选择信息增益最大的作为分裂树的特征属性
    3. 然后把原来样本的特征空间，按照分裂的规则，将其分为两个或多个更小的样本特征空间
    4. 如此往复，直到分裂完成，或者达到停止的条件
    
### 条件熵
**在信息增益中，加入了某个属性特征，导致信息熵发生了变化**   
**因此，我们引入了条件熵这个概念：**   
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/条件熵.jpg" width = 80% height = 80% />

### 信息熵使用示例

**还是刚才的数据集，我们利用信息熵计算公司来计算不同特征下的信息熵**   
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/信息熵示例1.jpg" width = 50% height = 50% />

**可以看出原系统的信息熵为1**   
**Age特征的条件熵为0.4**   
**由Age特征带来的信息增益为0.6**
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/信息熵示例2.jpg" width = 80% height = 80% />

**分别计算剩下两个特征的条件熵**   
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/信息熵示例3.jpg" width = 80% height = 80% />

**结论：**   
    Age的条件熵最小，带来的信息最大，所以根节点按照Age特征进行分裂  
    同时，在将原样本按照规则进行分割，拆分成子样本  
    然后再继续计算子节点的条件熵，直到分完
    由此，即可生成我们一开始画出的决策树
<img src="https://github.com/sfonly/Machine_Learning/blob/master/img_folder/Theory/DecisionTree/决策树1.jpg" width = 50% height = 50% />

## 常用决策树算法

未完待续...
### ID3
### C4.5
### CART
## 过拟合与剪枝
## 决策树的其他知识
### 分类树
### 回归树
### 决策树集成模型

