# 文章
1. Triple Cross-Domain Attention on Human Activity Recognition Using Wearable Sensors
2. DanHAR: Dual Attention Network for multimodal human activity recognition using wearable sensors
3. RepHAR: Decoupling Networks With Accuracy-Speed Tradeoff for Sensor-Based Human Activity Recognition
4. The Convolutional Neural Networks Training With Channel-Selectivity for Human Activity Recognition Based on Sensors
5. GRU-INC: An inception-attention based approach using GRU for human activity recognition
6. GRU with Dual Attentions for Sensor-Based Human Activity Recognition
7. A Novel Deep Multi‑Feature Extraction Framework Based on Attention Mechanism Using Wearable Sensor Data for Human Activity Recognition
8. A Hybrid Attention-Based Deep Neural Network for Simultaneous Multi-Sensor Pruning and Human Activity Recognition
9. A fuzzy convolutional attention-based GRU network for human activity recognition
10. Two-stream transformer network for sensor-based human activity recognition
11. Two-Stream Convolution Augmented Transformer for Human Activity Recognition
## 阅读要求
调研的总文章数大于30篇，将所有笔记、个人想法罗列在ideas.md

attention并不一定是学习上的注意力，~~很多文章特征提取+特征筛选也称为attention~~

1. <font color='red'>__摘要__</font>
2. <font color='green'>__介绍中与作者工作相关的部分、结论__</font>
3. 相关工作，提取解决的问题、 <font color='red'>~~**想法的来源**~~</font>
# 小结
此处插入图片

## 输入
* 数据集的选取
* 数据增强
## 模型
* 拓扑结构
    1. 静态残差
    2. 训练测试动态残差
* 架构的选取
    1. CNN
    2. LSTM
    3. GRU
    4. Transformer 编码器
* 注意什么
  

    输入[~~通道~~，时间步长，传感器模态]

    实际上输入为[批次，~~通道，~~窗口数，窗口长度，传感器轴]

    1. 在所有滑动窗口中，注意轴整体与时间的变化关系
    2. ~~在所有滑动窗口中，注意轴内部与时间的变化关系~~
    3. ~~在所有的滑动窗口中，注意轴整体与通道的变化关系~~
* <font color='red'>__如何注意__</font>

1. 在输入处注意
2. 在特征提取后注意
3. 先后注意
4. 双边注意，两条路
5. 残差注意
6. ~~三重注意~~

## 输出

效果都好