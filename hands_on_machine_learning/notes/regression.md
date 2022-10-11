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

## 多维线性回归
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
