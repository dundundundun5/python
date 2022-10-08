# regression

## 线性回归
1. 符号

    $$x\longrightarrow输入$$
    
    $$y\longrightarrow输出$$

    $$m\longrightarrow训练实例的个数$$

    $$(x, y)\longrightarrow一组训练实例$$

    $$(x^{(i)}, y^{(i)})\longrightarrow第i组训练实例$$

    $$\hat{y}\longrightarrow对y的预测值$$(0)

    $$f_{w,b}\big(x\big)=wx+b\rightarrow单变量线性回归$$
2.  平方误差代价函数

* 公式

$$ J(w,b)=\frac1{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2$$ (1)

由公式(0)可知 $\hat{y}^{(i)}=f_{w,b}(x^{(i)})=wx^{(i)}+b$

将$\hat{y}^{(i)}$代入公式则变成：

$$ J(w,b)=\frac1{2m}\sum_{i=1}^{m}({f_{w,b}(x^{(i)})}-y^{(i)})^2$$ (2)
* 目标

误差最小化 $\min{J(w,b)} $ 

## 梯度下降算法
用于最小化任何函数 $\min{J(w_1,w_2,w_3,...,w_n,b)} $

* 说人话

对于某个选定的点所表示的函数值，探索“当自变量变化时，函数值下降最快的变化路径
*  编程层面的算法（伪代码）

$$
while({w,b}未收敛) \\
            
tmp\_w=w-\alpha\frac{\partial{J(w,b)}}{\partial{w}}\\

tmp\_b=b-\alpha\frac{\partial{J(w,b)}}{\partial{b}}\\ 
w=tmp\_w \\
b=tmp\_b\\   
$$

其中$\alpha\longrightarrow学习率$，其控制梯度下降的速度，学习率太小导致梯度下降不明显，学习率过大导致梯度变化幅度过大从而无法获取最优解
* 批量梯度下降的概念

在梯度下降算法执行的过程中，计算包括所有实例$(\hat y^{(i)}, y^{(i)})$的代价函数
    
    


    