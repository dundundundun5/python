## lab

        gradient function for XX
## 梯度下降算法
用于最小化任何函数
$$\min{J(w_1,w_2,w_3,...,w_n,b)} $$

* 说人话

对于某个选定的点所表示的函数值，探索“当自变量变化时，函数值下降最快的变化路径
*  编程层面的实现（伪代码）
$$依据全体示例，分别对所有w_1,w_2,\cdots,w_n依次求偏导，算出他们各自的梯度下降后的值$$
$$repeat({\vec w,b}未收敛)\downdownarrows\\

w_j=w_j-\alpha\frac{\partial{J(\vec w,b)}}{\partial{w_j}}=w_j-\alpha(\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}w_j) \\

w_j\qquad1\leq{j}\leq{n}\\ \overbrace{w_1, w_2, \cdots, w_n} \\
\frac{\lambda}{m}w_j\longrightarrow 正则化项的求偏导后形态（see\quad in\quad regularization.md）\\
b=b-\alpha\frac{\partial{J(\vec w,b)}}{\partial{b}}=b-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})$$
其中$\alpha\longrightarrow学习率$，其控制梯度下降的速度，学习率太小导致梯度下降不明显，学习率过大导致梯度变化幅度过大从而无法获取最优解
* 正则化项的实际意义

    对以上$w_j$的式子进行移动变换得到：
    $$w_j=w_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}$$
    前一项相当于将$w_j$缩小一定比例，后一项保持一致，因此正则化实际上是在对参数作比例缩小
* 批量梯度下降的概念

在梯度下降算法执行的过程中，计算包括所有实例$(\hat y^{(i)}, y^{(i)})$的代价函数
* 检测收敛的方法

    建立代价函数和迭代次数的函数关系 
    $$J(\vec w,b)\rightleftarrows iterations $$
    同时从小到大调整学习率