# SklearnTests
基于Sklearn的典型数据挖掘应用

## [实验1 基于Sklearn的数据分类](https://github.com/dlvguo/SklearnTests/tree/master/Test1)

1.  [读取电费回收数据](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-1.py)

2. 电费回收数据预处理

   -  [电费回收数据规范化](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-2a.py)

   [电费回收数据特征数据提取方案1](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-2b1.py)

   ​	使用sklearn.feature_selection.VarianceThreshold：通过计算特征值的方差来提取特征，并通过variances_属性，查看各个特征的方差。

   ​	[电费回收数据特征数据提取方案2](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-2b2.py)

   ​	使用sklearn.feature_selection.GenericUnivariateSelec函数进行特征选择，可以设置不同的策略来进行单变量特征选择，也可以同时超参数调优选择最佳	单变量选择策略。该评估方法需要设置Target值，将IS_BAD（是否为欠费用户）设为Target值，计算各个特征值的分数。

3. 电费回收数据计算查准率、查全率、混淆矩阵与运行时间

   - [决策树](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-3-1.py)
   - [随机森林](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-3-2.py)
   - [神经网络](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-3-3.py)
   - [朴素贝叶斯](https://github.com/dlvguo/SklearnTests/blob/master/Test1/test1-3-4.py)

## [实验2  基于Sklearn的回归分析](https://github.com/dlvguo/SklearnTests/tree/master/Test2)

1.  [读取配网抢修数据](https://github.com/dlvguo/SklearnTests/blob/master/Test2/test2-1.py)

2.  配网抢修数据预处理

   - [配网抢修数据规范化](https://github.com/dlvguo/SklearnTests/blob/master/Test2/test2-2a.py)

   - [配网抢修数据特征数据提取方案](https://github.com/dlvguo/SklearnTests/blob/master/Test2/test2-2b.py)

     使用sklearn.feature_selection.GenericUnivariateSelec函数进行特征选择，可以设置不同的策略来进行单变量特征选择，也可以同时超参数调优选择最佳单变量选择策略。

3. 配网抢修数据计算MSE、RAE与运行时间

   - [随机森林](https://github.com/dlvguo/SklearnTests/blob/master/Test2/test2-3-1.py)
   - [神经网络](https://github.com/dlvguo/SklearnTests/blob/master/Test2/test2-3-2.py)
   - [线性回归](https://github.com/dlvguo/SklearnTests/blob/master/Test2/test2-3-3.py)

## [实验3  基于Sklearn的数据聚类](https://github.com/dlvguo/SklearnTests/tree/master/Test3)

1.  [读取移动客户数据](https://github.com/dlvguo/SklearnTests/blob/master/Test3/test3-1.py)

2.  移动客户数据预处理

   - [移动客户数据规范化](https://github.com/dlvguo/SklearnTests/blob/master/Test3/test3-2a.py)

   - [移动客户特征数据提取方案](https://github.com/dlvguo/SklearnTests/blob/master/Test3/test3-2b.py)

     使用sklearn.feature_selection.VarianceThreshold：通过计算特征值的方差来提取特征，并通过variances_属性，查看各个特征的方差。

3.  移动客户数据聚类质量分析并计算Silhouette、S_Dbw值

   - [K均值](https://github.com/dlvguo/SklearnTests/blob/master/Test3/test3-3-1.py)
   - [EM](https://github.com/dlvguo/SklearnTests/blob/master/Test3/test3-3-2.py)
   - [层次聚类](https://github.com/dlvguo/SklearnTests/blob/master/Test3/test3-3-3.py)

## [实验4 基于Sklearn的关联规则分析](https://github.com/dlvguo/SklearnTests/tree/master/Test4)

Sklearn中没有Apriori算法，采用[Apyori](https://github.com/ymoch/apyori)类库进行计算。看了相关资料对数据集进行离散化，泛化成[0-9]的标签（感觉数据处理方式以及计算方式有点问题），并对不同可信度之间计算关联集计算。

[代码链接](https://github.com/dlvguo/SklearnTests/blob/master/Test4/test4.py)
