## 正则化
* 用途
    
        正则化用于减小某些个特征对函数值的影响，来控制模型的拟合度
* 做法

        在原有的代价函数J后再加上一些关于参数w的项，形成新的代价函数。
        新的代价函数J'由两部分组成：原本的代价函数，参数w的项
* 原理

        在使用梯度下降算法最小化新的代价函数J'时，就需要平衡J和w的相对大小。
        通常会需要缩小参数w，从而达到让某些特征不再重要的目的
* 实现

    首先给出新的代价函数：
    $$J'(\vec w,b)=J(\vec w,b)+\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$
    对于线性回归来说：
    $$J'(\vec w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2
    +\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$
    $$f_{\vec{w},b}(\vec{x})=\vec{w}\vec{x}+b $$
    对于逻辑回归来说：
    $$J'(\vec w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2
    +\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$
    $$
    f_{\vec w,b}(\vec x)=\frac{1}{1+e^{(-\vec w\vec x+b)}}$$
    $$\lambda\longrightarrow正则化参数$$
