
<font size=6>
# 进展
1. IEEE Sensor Journal 仍然Under Review，LetPub都说四个月~七个月
2. 专利第二篇完成60%
3. 大论文进度需要加快
   
## 基于元学习和特征解耦的人类行为识别领域偏移研究
    
### 第1章 绪论
    1.1 人类行为识别的背景和问题  OK
    
    1.2 国内外研究现状 OK
    
    1.3 本研究的贡献 OK
### 第2章 [填入自建数据集的名称]数据集的采集与初步实验
    2.1 数据采集方法   采集方法已经确定，采用维特智能的低噪设备 -> 蓝牙 -> 上位机的现成软件
    
    2.2 数据初步实验方法
### 第3章 基于特征迁移的元学习框架
    3.1 问题定义 OK
    
    3.2 方法描述 方法框架已完成
    
    3.3 实验结果与可视化 实验中比较的部分有争议
    
    3.4 结论 跟随实验部分的变动而改变
### 第4章 基于反向激励的双分支CNN特征解耦
    4.1 问题定义 
    
    4.2 方法描述
    
    4.3 实验结果与可视化
    
    4.4 结论
### 第5章 总结与展望
    5.1 总结
    
    5.2 展望
参考文献
致谢

# meta-learning 元学习

调查笔记, 基于可穿戴传感器姿态估计的元学习框架目前有组在做，但是还没有用元学习框架尝试解决姿态估计中领域泛化问题的相关文章
## 介绍
1. 元学习最早提出于90年代，是一个抽象的概念，并没有实例，learn to learn是抽象概念的一个模糊概括
2. 2017年Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks由Chelsea Finn 1 Pieter Abbeel 1 2 Sergey Levine 1发表，第一位曾效力Google
 Brain，这篇文章成为了元学习在机器学习领域的第一个实例
3. 元学习和迁移学习的思想非常相似，但实验方法大不相同，所以元学习和迁移学习可以达成相同的目的，用相似但不同的手段

## Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks 
文章简称：MAML Model-Agnostic Meta-Learning

模型无关的 元学习

该框架用于少样本学习

记录MAML重点，便于回忆
### 前置


1. Task-support set-query set

    * task是元学习的基本单位，普通机器学习只把训练样本输入，元学习把task输入
    * task是任务，一个任务由support set支持集和query set查询集组成。前者用于训练，后者用于验证，也许因为是一种为了少样本学习的框架，把数据划分的名称换了名字
2. N-ways K-shots 

    N-K是task的配置。N类，每类有K个样本，分类任务中的N类
3. 简言之元学习的数据划分

    * 数据划分为大训练集和测试集
    * 每次训练都从大训练集中任取K个support样本和任意个query样本，一个任务由 K个support和任意个query组成。
    * 机器学习中的一条数据~元学习中的一个任务
    * 机器学习把大训练集划分为小训练集和验证集，元学习把大训练集划分为若干个小小训练集和小小验证集（support set + query set）
### MAML算法一言蔽之

1. MAML寻求的是更有泛化能力的初始化权重，learn to learn在此处可以等效于learn to learn the best initial weights
2. MAML不急于让原模型在跟着小小训练集进行梯度更新。梯度依然会更新，但不是立刻在原模型上生效。
3. MAML希望原模型在小小训练集上的梯度更新也能更好地降低在小小验证集上的损失，永远在梯度更新前向前看一步。

### MAML算法详解
Require: 数据划分为任务集合T，T含有M个任务t，模型f，两个学习率超参数$\alpha,\beta$

1. 随机初始化模型参数$\theta$, 设置$batch\space size$
2. $While\space i\in迭代轮次:$
    
    1. 采样$batch\space size$个任务:$t\in T$
    2. $For\space t_i\in [t_1,t_2,\cdots,t_{batch\space size}]:$

        1. 用$t_i$的小小训练集中的N类，每类K个样本计算平均loss，计算梯度下降后的梯度$\theta_i$,但并不更新到原模型
        $$\theta_i=\theta-\alpha\nabla_{\theta}\mathcal{L}_{t_i}(f_{\theta})$$
        2.  复制模型，并把$\theta_i$用于复制的模型。在复制的模型上，$t_i$的小小验证集中的N类，每类任意个样本计算平均loss。
        $$\sum_{i=1}^{batch\space size}\mathcal{L}^{'}_{t_i}(f_{\theta_i})$$

    3. 原模型用于更新参数的loss为上述任务loss的总和，此时才有了原模型参数的梯度下降
    $$\theta=\theta-\beta\nabla_{\theta}\sum_{i=1}^{batch\space size}\mathcal{L}^{'}_{t_i}(f_{\theta_i})$$

MAML算法完毕，实际是用空间复杂度换收敛时间。从而依靠若干份子任务的表现，来判断未来是否可期。

由于计算梯度时存在二阶偏导数，原文使用了近似为一阶导数的方法，原文MAML也有First order approximation的叫法

## Learning to Generalize: Meta-Learning for Domain Generalization

### MLDG 简介

1. MLDG Meta-Learning for Domain Generalization 是MAML元学习框架在DG训练测试上的适配
2. 原文设计了DG和RL两大领域的元学习适配，并在拟合图像曲线、目标检测、倒立摆cart-pole、越野车mountain car开展实验
3. 数据集按域划分为训练集，验证集，测试集。
4. MLDG不同于MAML，损失函数不同

### MLDG 算法查漏补缺
1. meta-train、meta-test分别从训练集、验证集中采样
1. 损失函数：$\argmin{\mathcal{F}(\Theta)+\beta\mathcal{G}(\Theta-\alpha\mathcal{F}'(\Theta))}$，前者是模型在meta-train上的损失，后者是原模型根据meta-train损失梯度下降一次后得到的参数，这个参数在meta-test上的损失
2. 作者使用了一阶泰勒在$x=\Theta_0$处展开$\mathcal{G}(x)$。

$$\mathcal{G}(x)=\mathcal{G}(\Theta_0)+\mathcal{G}'(\Theta_0)\times(x-\Theta_0)$$

令$x=\Theta-\alpha\mathcal{F}'(\Theta)$

所以$x-\Theta=-\alpha\mathcal{F}'(\Theta)$

让$\mathcal{G}(x)在x=\Theta处展开$

$$
\begin{align}
    \mathcal{G}(\Theta-\alpha\mathcal{F}'(\Theta))&=\mathcal{G}(x)\\
    &=\mathcal{G}(\Theta)+\mathcal{G}'(\Theta)\cdot(x-\Theta)\\
    &=\mathcal{G}(\Theta)+\mathcal{G}'(\Theta)\cdot(-\alpha\mathcal{F}'(\Theta))
\end{align}
$$
3. 损失函数变成
$$
\begin{align}
   \argmin{\mathcal{F}(\Theta)+\beta\mathcal{G}(\Theta-\alpha\mathcal{F}'(\Theta))}&=\argmin{\mathcal{F}(\Theta)+\beta\mathcal{G}(\Theta)-\alpha\beta\mathcal{G}'(\Theta)\cdot\mathcal{F}'(\Theta)}\\
   &\leq \argmin{\mathcal{F}(\Theta)+\beta\mathcal{G}(\Theta)-\frac{\alpha\beta\mathcal{G}'(\Theta)\cdot\mathcal{F}'(\Theta)}{\Vert\mathcal{G}'(\Theta)\Vert_2\Vert\mathcal{F}'(\Theta)\Vert_2}}
\end{align}


$$

4. 前者表示最小化同一套参数在meta-train、meta-test的损失，后者表示最大化$\mathcal{G}'(\Theta)$和$\mathcal{F}'(\Theta)$二者的余弦相似度
5. $\mathcal{G}'(\Theta)$和$\mathcal{F}'(\Theta)$表示同一套参数在meta-train、meta-test上的梯度，最大化点积表示希望二者梯度的余弦相似度最大，$\cos0\degree=1$，则希望二者梯度尽可能同向
6. 后者要求梯度尽可能同向，表示参数优化方向需要同时考虑meta-train和meta-test，从而鼓励学习能帮助模型学习领域转移的元知识 （就是迁移学习的迁移知识换皮），而不是仅在meta-train上过拟合


## Feature-Critic Networks for Heterogeneous Domain Generalisation

### 算法简介
1. 标题带有Feature-Critic，表示损失函数是和隐特征变量相关，直译就是特征评价
2. 特征评价，仍然基于原参数和梯度下降后的参数，前者用于最小化传统loss，后者用于最小化辅助loss

### 算法查缺补漏
1. meta-train、meta-test分别从训练集、验证集中采样
2. 特征提取网络$f_{\theta}$，分类器$g_{\phi}$，全域集合$\mathcal{D}$，学习率$\alpha，\eta$
3. 损失函数$l_{train}+l_{aux}$为

$$\begin{align}
    &\argmin_{\theta,\phi}{\sum_{D_j\in\mathcal{D_{train}}}\sum_{d_j\in D_j}CE(g_{\phi}(f_{\theta}(x_j)),y_j)} + \\ 

    &\argmin_{\omega}{\frac{1}{M}\sum_{i=1}^M MLP_{\omega}(f_{\theta}(x_j)_i)}
\end{align}$$
4. $\theta^{OLD}=\theta-\alpha\nabla_{\theta}l_{train}$
5. $\theta^{NEW}=\theta^{OLD}-\alpha\nabla_{\theta}l_{aux}$ 只有该参数含有因$\omega$导致的参数下降
6. 损失函数$l_{meta}$为

$$\begin{align}
    &\argmin_{\omega}{\sum_{D_j\in\mathcal{D_{valid}}}\sum_{d_j\in D_j}\tanh{(CE(g_{\phi}(f_{\theta}^{NEW}(x_j)),y_j)-CE(g_{\phi}(f_{\theta}^{OLD}(x_j)),y_j))}}
\end{align}$$

7. 最终的梯度下降为

$$\begin{align}
    \theta&=\theta-\eta(\nabla_{\theta}l_{train}+\nabla_{\theta}l_{aux})\\
    \phi&=\phi-\eta\nabla_{\phi}l_{train}\\
    \omega&=\omega-\eta\nabla_{\omega}l_{meta}
\end{align}$$

8. 由MLP构成的特征评价网络，输入特征图得到的损失被用于进行临时的参数更新，而临时参数被用在meta-test从而期望“参数的每一次下降都有利于领域转移”，问题在于$l_{meta}$是$\theta$的函数，怎么对$\omega$求导？
9. <font color='red'>这篇也有一个未核实的问题。</font>
在 https://zhuanlan.zhihu.com/p/146877957《小王爱迁移》系列之二十四：元学习的前世今生

用户（https://www.zhihu.com/people/song-74-79-76）：

__博主您好，我近期看了一篇元学习做领域泛化的文章，叫“ Feature-Critic Networks for Heterogeneous Domain Generalisation”。然而，我认为文章中构造的优化函数（公式5）是不可导的，不知道您了解吗？__

博主（https://www.zhihu.com/people/jindongwang）：

__那个我看过，他代码能跑通，就很奇怪[捂脸]__


## MetaReg: Towards Domain Generalization using Meta-Regularization
1. 网络架构不一样，F是特征提取网络，$T_1~T_p$是p个领域各自的分类网络（全连接层）
2. p个领域各自的分类网络注定学到p个领域各自的隐藏层表示
3. 作者希望用正则化函数，促使p个不同网络学到同一种隐藏层表示
### 算法简介
已知：
* 特征提取网络$F_\psi$,分类网络$[T_{\theta_1},\cdots,T_{\theta_p}]$，数据集$D$
* $D_{train}\cup D_{test}=D$,$D_{meta\_train}\cup D_{meta\_valid}=D_{train}$, $D_{train}$有p个领域,$D_{test}$作为测试集，随意。
* **正则化函数$R_\phi(\theta)=\sum_i\phi_i|\theta_i|$**
1. While $i\in 迭代轮次$:
    * i=1时正则化函数的参数为$\phi_1$
    1. While $j\in k$:
       * 损失函数为交叉熵$\mathcal{L}_1$=CrossEntropy(*,\*)
       * 全模型一共$k$步梯度下降
    1. 得到参数$[\psi^{(k)},\theta_1^{(k)},\cdots,\theta_p^{(k)}]$
    2. 任取$a,b\in[1,2,\cdots,p]$两个领域的数据4. a作为meta_train,b作为meta_valid
    3. $\beta_1\gets\theta_a^{(k)}$
    4. 用领域a的一部分或所有数据
    5. While $j\in [2,\cdots,l]$:
        * $\beta_j=\beta_{j-1}-学习率1\times\nabla_{\beta_{j-1}}[\mathcal{L}_1(\psi^{(k)},\beta_{j-1})+R_\phi(\beta_{j-1})]$
        * 正则化函数含义为多个分类网络参数矩阵的模的加权和
        * 得到$\beta_l$
    6. 使用领域b的一部分或所有数据
        
        * $\phi_2=\phi_1-学习率2\times\nabla_{\phi}\mathcal{L}_1(\psi^{(k)},\beta_l)|_{\phi=\phi_1}$
        * 
__只有正则化在宏观控制p个分类网络的参数差异性，该方法元学习套壳正则化__

## METANORM: LEARNING TO NORMALIZE FEW-SHOT BATCHES ACROSS DOMAINS

粗看，走的是领域泛化+批量归一化，然而批量归一化仍然未来可期，相关工作少，已发表工作的时间跨度大

## Domain Generalization via Model-Agnostic Learning of Semantic Features


## Shape-Aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains
是一个磁共振图像分割任务的方法，其在原有的元学习框架上增加了两个损失函数的部分


### 算法总结
$$
\begin{align}
    \mathcal L_{meta}=\mathcal{L}_{seg}+\lambda_1\mathcal{L}_{compact}+\lambda_2\mathcal{L}_{smooth}
\end{align}
$$

1. $\mathcal{L}_{seg}$是图像分割必需的损失函数
2. $\mathcal{L}_{compact}$用于解决分割线的紧凑程度
   
    1. 借助物理知识IsoPerimetric Quotient等周商
    2. 等周商$C_{IPQ}=\frac{P^2}{4\pi A}$ P周长A面积
    3. 图像分割线组成闭合图形的等周商越大，图形越紧凑
3. $\mathcal{L}_{smooth}$用于解决分割线的平滑程度
   
    1. 计算不同领域中每个样本的隐藏向量的距离
    2. 试图达到同一类别的分割线的形状相似，不同类别的分割线的形状有差异
4. 算法是领域知识风格的正则化+普适的元学习，作为一个任务的专用对策，不具备领域间通用性
## (NOT_READY)GENERALIZING ACROSS DOMAINS VIA CROSS-GRADIENT TRAINING