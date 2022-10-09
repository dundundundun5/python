# regression

## 单维线性回归
1. 符号

    $$x\longrightarrow输入$$
    
    $$y\longrightarrow输出$$

    $$m\longrightarrow训练实例的个数$$

    $$(x, y)\longrightarrow一组训练实例$$

    $$(x^{(i)}, y^{(i)})\longrightarrow第i组训练实例$$

    $$\hat{y}\longrightarrow对y的预测值$$(0)
    $$\hat y^{(i)}=wx^{(i)}+b\longrightarrow对第i个示例的预测值 $$
    $$f_{w,b}\big(x\big)=wx+b\rightarrow单维线性回归$$
2.  平方误差代价函数

* 公式

$$ J(w,b)=\frac1{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2$$ (1)

由公式(0)可知 $\hat{y}^{(i)}=f_{w,b}(x^{(i)})=wx^{(i)}+b$

将$\hat{y}^{(i)}$代入公式则变成：

$$ J(w,b)=\frac1{2m}\sum_{i=1}^{m}({f_{w,b}(x^{(i)})}-y^{(i)})^2$$ (2)
* 目标

误差最小化 $\min{J(w,b)} $ 

# 多维线性回归
1. 符号
   
    $$n\longrightarrow特征个数 $$
    $$\vec{x}^{(i)}\longrightarrow第i个示例的特征向量$$
    $$x_{j}^{(i)}\longrightarrow第i个示例的第j个特征 $$
    $$\vec{w}=\left[w_1,w_2,w_3,...,w_n\right]\longrightarrow参数向量$$
    $$\vec{x}=\left[x_1,x_2,x_3,...,x_n\right]\longrightarrow特征向量$$
    $$f_{w,b}(x)=w_1x_1+w_2x_2+\cdots+w_nx_n\longrightarrow多维线性回归$$(3)
    将式子看成向量的点乘，则可以写为：
    $$f_{\vec{w},b}(\vec{x})=\vec{w}\vec{x}+b \longrightarrow多维线性回归$$(4)
     $$\hat{y}\longrightarrow对y的预测值$$
    $$\hat y^{(i)}=\vec w\vec x^{(i)}+b\longrightarrow对第i个示例的预测值 $$
2. 平方误差代价函数

* 公式

    $$J(\vec w, b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2

     $$
## 向量化
上面介绍了多维线性回归的点乘表示：
* 伪代码

    ```f=np.dot(w,x)+b```
## 梯度下降算法
用于最小化任何函数
$$\min{J(w_1,w_2,w_3,...,w_n,b)} $$

* 说人话

对于某个选定的点所表示的函数值，探索“当自变量变化时，函数值下降最快的变化路径
*  编程层面的算法（伪代码）
$$依据全体示例，分别对所有w_1,w_2,\cdots,w_n依次求偏导，算出他们各自的梯度下降后的值$$
$$repeat({\vec w,b}未收敛)\downdownarrows\\

w_j=w_j-\alpha\frac{\partial{J(\vec w,b)}}{\partial{w_j}}=w_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)} \\
w_j\qquad1\leq{j}\leq{n}\\ \overbrace{w_1, w_2, \cdots, w_n} \\
b=b-\alpha\frac{\partial{J(\vec w,b)}}{\partial{b}}=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})$$
其中$\alpha\longrightarrow学习率$，其控制梯度下降的速度，学习率太小导致梯度下降不明显，学习率过大导致梯度变化幅度过大从而无法获取最优解

* 批量梯度下降的概念

在梯度下降算法执行的过程中，计算包括所有实例$(\hat y^{(i)}, y^{(i)})$的代价函数
* 检测收敛的方法

    建立代价函数和迭代次数的函数关系 
    $$J(\vec w,b)\rightleftarrows iterations $$
    同时从小到大调整学习率
## 正规方程
用于求解线性多项式系数的一种方法，无笔记
## 特征缩放（归一化）
* 原因

    将数值范围不同的特征放在一起，那么数值范围相对较大的特征值在拟合过程中系数选取的难度更高，一旦参数作微小的改变，就会大大影响预测结果，而数值范围相对较小的特征值的系数则需要大幅度改变才能影响到预测结果
* 时机

    最好是-1到+1之间，大一些也没关系，太大或太小都要考虑缩放
* 最大值缩放

    对第j个特征：
$$x_j = \frac{x_j}{\max}$$
* 均值缩放

对第j个特征，算得其平均值：
$$x_j = \frac{x_j-\mu_j}{\max-\min}$$  
* z-score缩放

对第j个特征，算得其平均值和标准差：
$$x_j = \frac{x_j-\mu_j}{\sigma_j}$$ 
## 特征工程
筛选合适的特征、根据已知特征创建可能需要的特征
## 多项式回归
使用某一个特征的幂次项，从而更好地拟合训练集数据
