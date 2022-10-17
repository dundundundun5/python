# neural network
人工神经网络的简要笔记，具体需依靠书籍
## 基础
* 组成

    神经网络由输入层、隐藏层、输出层组成。
    其中，输入层包含需要输入的特征，隐藏层和输出层分别包含了至少一个神经元
* 神经元

    神经元可视为一个处理模块，输入是一套特征的线性组合，输出称为激活activation，处理的数学逻辑称为激活函数activation function
* 运作方式

    每一层的输出都是下一层的输入，最后一层的输出是分类结果
* 区别

    神经网络相对于逻辑回归的最大的区别是：逻辑回归依据给定的特征或者特征之间线性组合后得到新的特征作分类预测，而神经网络不手动获取隐含的新特征，而是通过机器学习的方式获取

* 激活函数
  
    激活函数的主要作用是完成数据的非线性变换，解决线性模型的表达、分类能力不足的问题;
    
    因为如果使用线性的激活函数，那么输入x跟输出y之间的关系为线性的，便可以不需要网络结构，直接使用线性组合便可以。
* 符号

    $$\vec x \longrightarrow 输入特征向量$$
    $$\vec a^{[i]}_{j} \longrightarrow 第i层第j个激活向量$$
    $$z\longrightarrow输入的线性回归组合 $$
    $$g(z)\longrightarrow激活函数的笼统表示$$
* 前向传播

    指每一层神经元的输入都由前一层的输出提供、
## Tensorflow
吴恩达课程中介绍了少量的tensorflow的实践内容，也补充了数学的基础
1. tensorflow的数据类型

    1. tensorflow的所有输入均为[[列表1],[列表2],...,[列表n]]
    2. numpy的输入可以是列表，要注意定义数据格式为二维列表
2. 前向传播的实现

    
    ```
    循环实现：
    def dense(a_in,W,b,g):
        units = W.shape[1]
        a_out = np.zeros(units)
        for j in range(units):
            w = W[:,j]
            z = np.dot(w,a_in) + b[j]
            a_out[j] = g(z)
        return a_out
    ```
    ```
    向量化实现：
    def dense(A_in,W,B):
        Z = np.matmul(A_in,W) + B
        A_out = g(Z)
        return A_out
    ```
    向量化的优势：矩阵乘法可由硬件计算，速度快很多
3. 矩阵基础

    详见《张宇线代9讲》，在考研数学方面下那么多功夫真不是乱来的
## 激活函数的类别
1. sigmoid

    $$z=\vec{w}\cdot\vec{x}+b $$
    $$a1=g(z)=\frac{1}{1+e^{-z}}=P(y=1|\vec{x}) $$
2. ReLU(Rectified Linear Unit)

    $$z=\vec{w}\cdot\vec{x}+b $$
    $$g(z)=max(0,z) $$
3. softmax

    用于多酚类
## Multi-class Classification
指输出标签的种类不再是两种，而是多种
1. softmax详解

    $$z_j=\vec{w_j}\cdot\vec{x}+b_j\quad j=1,\cdots,N $$
    $$a_j=g(z_j)=\frac{e^{z_j}}{\sum_{k=1}^{N}e^{z_k}}=P(y=j|\vec{x}) $$
    $$softmax用于多分类：图形分类（3种）\\
    z_1=\vec{w_1}\cdot\vec{x}+b_1\quad 
    a_1=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}\\
    z_2=\vec{w_2}\cdot\vec{x}+b_2\quad 
    a_2=\frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}\\ 
    z_3=\vec{w_3}\cdot\vec{x}+b_3\quad
    a_3=\frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}\\    
    $$
2. 损失函数

    $$loss(a_1,a_2,\cdots,a_N,y)=\begin{cases}
        -\log a_1\quad y=1\\
        -\log a_2\quad y=2\\
        \qquad\qquad\vdots\\
        -\log a_N\quad y=N\\
    \end{cases} $$-\log a_1\quad y=1\\
3. softmax的改进

        由于计算机双精度的阶码和尾数在一系列计算中可能自动舍入，现不再要求模型一步到位。
        取而代之，在输出层先计算出z，再手动书写代码计算softmax函数的值
        这样可以提高数值的表示精度
4. multi-class和multi-label

        前者指代一次输出一个标签，标签有多种
        后者指代一次输出多个标签，标签也有多种
5. Adam(Adaptive Moment estimation)

    是一种类似梯度下降的最小化算法
    特点在于对所有w，以及一个b，施加不同的学习率，来逐步计算最低误差
    1. 如果$w_j或b$在误差函数的值上始终朝一个方向移动，则增大$a_j$
    2. 如果持续震荡，则减小$a_j$
## 卷积神经网络
对于特征的选择不再是线性选择，而是用滑动窗口的方式，目前无法理解