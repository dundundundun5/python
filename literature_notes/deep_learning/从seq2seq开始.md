<font size=5>

# Seq2Seq
Seq2Seq用于最早的机器翻译

编码器是一个RNN，可以是双向RNN（看到整个句子），
解码器用另一个RNN

图片
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101180958.png)
编码器是没有输出的RNN

编码器最后时间步的隐藏状态用作解码器的初识隐藏状态

训练和预测，流程不一样
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101181035.png)
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101181054.png)
早期是seq2seq，然而如今已经由BERT GPT
衡量标准 BLEU

$$ p_n是预测所有n-gram的精度\\
n-gram是n个连续的词$$

标签是My name is hello world
预测是name is hi hello

查看预测的n-gram是否存在于标签中的古典概率

$$p_1 = \frac{name,is,hello}{name,is,hi,hello}=\frac{3}{4}$$

$$p_2 = \frac{name-is, is-hello}{name-is, is-hello, hi-hello}=\frac{2}{3}$$

$$p_3 = \frac{无}{name-is-hi, is-hi-hello}=0$$


$$BLEU=e^{\min{(0,1-\frac{标签长度}{预测长度})}}\prod_{n=1}^kp_n^{\frac{1}{2^n}} $$

标签比预测长很多，那么指数的负数次方将会映射到很小的数字，表示对预测过短的惩罚，因为越短越容易命中精度

连乘带有分数次方，因此越长的匹配具有越高的权重

$$比如p_{10}=\frac{1}{4} 则p_{10}^{\frac{1}{2^{10}}}=0.25^{\frac{1}{1024}}具有高权重$$

# Attention

心理学，随意线索：人为关注的特征
不随意线索：物体本源的特征

注意力机制有 查询和键值对
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101181126.png)
Query（查询）是随意线索，Key是不随意线索，Value的含义看具体情况，kv可以一样可以不一样

通过注意力池化层来有偏向性地选择某些输入

不像卷积会提取不随意线索

## 影子 Nadaraya-Watson核回归

$$给定数据(x_i,y_i),i=1,\cdots,n \\
对于任意x 均给出f(x)=\frac{1}{n}\sum_iy_i$$
非参核回归是
$$f(x)=\sum_{i=1}^n\frac{核(x-x_i)}{\sum_{j=1}^n核(x-x_j)}y_i $$

有一堆数据，来个新数据，就用相近的数据去查询，非参

如果使用高斯核=$\frac{1}{\sqrt{2\pi}}e^{-\frac{u^2}{2}}$
则
$$f(x)=\sum^n_{i=1}\frac{\frac{1}{\sqrt{2\pi}}e^{-\frac{{(x-x_i)}^2}{2}}}{\sum_{j=1}^n\frac{1}{\sqrt{2\pi}}e^{-\frac{{(x-x_j)}^2}{2}}}y_i $$
$$ f(x)=\sum^n_{i=1}softmax\bigg(-\frac{1}{2}(x-x_i)^2\bigg)y_i$$
softmax给了0~1之间的值，这些值加起来等于1，作为权重乘到各个y上面

带参的核回归就变成
$$ f(x)=\sum^n_{i=1}softmax\bigg(-\frac{1}{2}\big((x-x_i)w\big)^2\bigg)y_i$$

一般可以写作$f(x)=\sum_i\alpha(x,x_i)y_i$，这里的$\alpha$通常称为注意力权重
## 常用的注意力权重

上面提到的$\alpha$叫做注意力权重$f(x)=\sum_{i} \alpha(x,x_i)y_i=\sum_isoftmax\bigg(-\frac{1}{2}(x-x_i)^2\bigg)y_i$

在softmax里的函数是注意力分数函数,记作A
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101181148.png)

拓展到高维度

假设query和m对key-value均为向量，简称为$\vec q,\vec k, \vec v$，后续用加粗表示，它们的维度分别为q，k，v
则$f(\pmb q,(\pmb{k_1},\pmb{v_1}),\cdots,(\pmb{k_m},\pmb{v_m}))=\sum_{i=1}^m\alpha(\pmb{q},\pmb{k_i})\pmb{v_i}$，其中$\alpha(\pmb{q},\pmb{k_i})=softmax(A(\pmb{q},\pmb{k_i}))$

A是注意力分数函数，A的设计决定了注意力的偏向性

## additive attention

可学参数 $\pmb{W_q}(h,q)$，$\pmb{W_k}(h,k)$，$\pmb{v}(h,1)$

$$A(\pmb{k},\pmb{q})=\pmb{v}^Ttanh(\pmb{W_kk}+\pmb{W_qq})$$

$\pmb{W_k}(h,k)*\pmb{k}(k,1)=\pmb{W_kk}(h,1)$

$\pmb{W_q}(h,q)*\pmb{q}(q,1)=\pmb{W_qq}(h,1)$

$\pmb{v}^T(1,h)*tanh(\pmb{W_kk}+\pmb{W_qq})(h,1)=注意力分数(1)$

相当于是在合并kq以后将其放入一个隐藏大小为h输出大小为1的单隐藏层mlp

$\pmb{W_kk}+\pmb{W_qq}$相当于大小为h的隐藏层，输出层的大小为1，输入是k和q

好处是k qv可以是任意长度

矩阵化结果，正在学习
## scaled dot-product attention

如果q和k都是同样d长度，那么可以用相似度去衡量q和k的距离

$$A(\pmb{q},\pmb{k_i})=\frac{\lang{\pmb{q},\pmb{k_i}}\rang}{\sqrt{d}}$$

矩阵化版本
$\pmb{Q}(n,d)$，$\pmb{K}(m,d)$，$\pmb{V}(m,v)$

注意力分数为$A(\pmb{Q},\pmb{K})=\frac{\pmb{Q}\pmb{K}^T}{\sqrt{d}}(n,m)$

对于n个query，每个query在每个key上的注意力分数，每行有m个注意力分数，分别由query和每个key

注意力池化汇聚$f=softmax(A(\pmb{Q},\pmb{K}))\pmb{V}(n,v)$

最后得出，对于n个query，每个query都会得到一个长为v的向量，这个向量是众多向量$\pmb{v_1},\cdots,\pmb{v_m}$根据注意力权重的加权和

## 注意力融入seq2seq
任务不一样，注意力的融入方式完全不一样，此处学习仅供理解

My name is dundun-> 我的 名字 是 顿顿

假设语序是相似的，那么动机是，预测翻译的下一个词时，应该把目光聚焦于源句子的对应意思的词，而seq2seq的预测总是基于源句子的最后一个词的编码器隐藏状态

编码器对每个词的输出作为key和value

解码器RNN对上一个词的输出作为query，统一语义空间

注意力的输出和下一个词的词嵌入合并进入rnn层

最开始是没有attention的，训练的过程构造了attention，预测的过程使用了attention

# self-attention
自注意力的自，意思是key、value、query全都源于自己

$\pmb{x_1},\cdots,\pmb{x_n} \in (d, 1)$

自注意力池化汇聚层将$x_i$当作key，value，query来对序列抽取特征得到$\pmb{y_1},\cdots,\pmb{y_n} \in (d, 1)$

$\pmb{y_i}=f(\pmb{x_i},(\pmb{x_1,x_1}),\cdots,(\pmb{x_n},\pmb{x_n}))$
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101181203.png)
## 位置编码
自注意力是没有位置信息的

在输入中嵌入位置信息

假设长度为n的序列是$\pmb{X}(n,d)$,那么使用位置编码矩阵$\pmb{P}(n,d)$来输出$\pmb{X}+\pmb{P}$作为自编码输入

$$P_{i,2j}=\sin(\frac{i}{10000^{\frac{2j}{d}}})$$
$$P_{i,2j+1}=\cos(\frac{i}{10000^{\frac{2j}{d}}})$$

偷图
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20230101181802.png)
0的二进制是：000
1的二进制是：001
2的二进制是：010
3的二进制是：011
4的二进制是：100
5的二进制是：101
6的二进制是：110
7的二进制是：111

该位置信息呈现的还是相对位置

$i+\delta\longrightarrow i$（两个词在句子中的相对位置）
A B C D E

B D 2 4

B C D E 

B D 1 3 
令$\omega_j=\frac{1}{10000^{\frac{2j}{d}}}$则$p_{i,2j}=\sin{(\omega_j)}\quad p_{i,2j+1}=\cos{(\omega_j)}$
$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

完全并行，任何一个输出，都能看到整个序列的信息，但是计算复杂度高

## Transformer

Transformer也是编码器解码器的架构，然而Transformer是一个
纯基于自注意力的结构，不再具有RNN

## multi-head attention