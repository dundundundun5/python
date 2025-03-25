# 单隐藏层LSTM笔记

图中的X代表按位置作乘法
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20221218125627.png)
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20221218133904.png)
## 公式


假设x是5维列向量，LSTM中每个门里隐藏层神经元个数为7，则$h_{t-1}$是7维列向量
1. 遗忘门

    $$f_t=\sigma(W_f\cdot[
        \begin{matrix}
            h_{t-1}\\x_t
        \end{matrix}
        
    ] + b_f) $$

2. 输入门

    $$i_t=\sigma(W_i\cdot[
        \begin{matrix}
            h_{t-1}\\x_t
        \end{matrix}
        
    ] + b_i) $$
    $$c_{t-1}'=\tanh(W_c\cdot[
        \begin{matrix}
            h_{t-1}\\x_t
        \end{matrix}
        
    ] + b_c) $$

    
3. 输出门

    $$o_t=\tanh(W_o\cdot[
        \begin{matrix}
            h_{t-1}\\x_t
        \end{matrix}
        
    ] + b_o) $$
    $$h_t=o_t*\tanh(c_t) $$
4. 维度

    $$W_{系列}\longrightarrow(隐藏=7，隐藏+x维度=7 + 5)$$
    $$[\begin{matrix}h_{t-1}\\x_t\end{matrix}]\longrightarrow(隐藏 +x维度=7+5,1) $$
    $$b_{系列}\longrightarrow(隐藏=7，1) $$
    $$中间的输出\longrightarrow(隐藏=7，1)$$

# GRU笔记

![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20221218133557.png)
![](https://raw.githubusercontent.com/dundundundun5/pictures/main/formal20221218134025.png)
## 公式
假设x是5维列向量，GRU中每个门里隐藏层神经元个数为7，则$h_{t-1}$是7维列向量
1. 更新门

    $$z_t=\sigma(W_z\cdot[
        \begin{matrix}
            h_{t-1}\\x_t
        \end{matrix}
        
    ]) $$
    
2. 重置门

    $$r_t=\sigma(W_r\cdot[
        \begin{matrix}
            h_{t-1}\\x_t
        \end{matrix}
        
    ]) $$
3. 结果
    $$h_{t-1}'=\tanh(W_h\cdot[
        \begin{matrix}
            r_t*h_{t-1}\\x_t
        \end{matrix}
        
    ]) $$

    $$h_t=(1-z_t)*h_{t-1}+z_t*h_{t-1}' $$
4. 重点

    <font color='red'>重置门的值越小，流入信息越少/先前信息遗忘得越多，更新门的值越大，流入信息越多</font>