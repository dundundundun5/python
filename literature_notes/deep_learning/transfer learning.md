# Transfer Learning

<font color='red'>__这篇笔记是综述5的阅读笔记，论文直接从第9页开始看__</font>

* 迁移学习+基于传感器数据的人类活动识别 HAR

    9篇综述疯狂筛选出来的1篇综述

1. 20250328 A comprehensive survey of transfer dictionary learning

    Neurocomputing的一篇好综述，迁移学习和字典学习的结合，由于对字典学习完全不知道，这个综述如果好奇的话可以看看
2. 20250221 A review of recent advances and strategies in transfer learning

    这篇综述还是偏方法汇总，没有什么很独到的见解
3. 20250108 Application of human activity/action recognition: a review

    Multimedia Tools and Application Q4这篇看标题是应用，其实就是描述整个过程，带了点深度学习但不多，介绍了一些特别基础的模型，避雷
4. 20250101 Generalization in neural networks: A broad survey
    
    Neurocomputing的一篇好总数，以为是迁移学习，其实还是讲模型在测试集上的泛化能力，没有迁移学习的概念，但是也讲了很多基础创新的动机的方法论，可选看
5. <font color='red'>20250213 Transfer Learning in Sensor-Based Human Activity Recognition: A Survey</font>

    ACM Computerer Survey的一篇深度好文，必看，位于次要优先级
6. 20150801 Semantic human activity recognition: A literature review

    Pattern Recognition的一篇技术性综述，老文章了，涉及语义，难度是断崖式领先的，想研究可以走这个，目前全是开放题
7. 20221001 Human activity recognition using tools of convolutional neural networks: A state of the art review, data sets, challenges, and future prospects

    Computers in Biology and Medicine 太入门了，不看

8. <font color='red'>20230801 Generalizing to Unseen Domains: A Survey on  Domain Generalization</font>

    IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING，深度好文，有基础想找具体方法了可以看，找对比工作也许，23年的也不老，必看，位于首要优先级
9.  <font color='blue'>20230402 Domain Generalization: A Survey</font>

    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 同为23年的一篇好综述，必看，至少能了解计算机视觉的迁移学习曾经火过哪些内容，

    HAR版本是落后的，计算机视觉2018年爆火的方法论在2022年的HAR爆火了

## 一些问题


其实迁移学习在计算机视觉和自然语言处理已经很成熟了，你可以方法直接拿过来用，但是一定不会好用，因为人类活动识别存在一些特有的问题。

别的领域的方法拿过来肯定要根据这些问题改进


### 缺乏标记好的数据
* 不是缺乏数据，而是缺乏 __大量__ 标记好的数据

    目前数据集很多很杂，
    
    ```CASAS, UCI-HAR, Mobiact, PAMAP2, OPPORTUNITY,DSADS```
    
    不缺数据集的个数，但是HAR没有一个像ImageNet那样统治力极强的数据集
* 数据集又多又杂不是坏事，采集标准不统一是坏事

    目前HAR的迁移学习还是很怕数据集的原生问题，采样频率、佩戴位置和朝向。大家都不一样。CIFAR10和100可都是图片，图片再怎么变都是像素点，甚至分辨率都一样
### 噪声大

* 传感器的噪声很巨大

    噪声巨大，并不意味着和当前的活动有关或无关，要合理地去解决
* 目前噪声的处理仅仅在于最初的数据清洗

### 很难学到通用的潜在特征
* 通用是多个数据集的同一个活动的潜在特征

    每个数据集采集标准都不一样，不可能简简单单学到通用的潜在特征，不像图片，图片就算翻天覆地也还是像素点，传感器位置朝向一变，拿到的数据就完全不一样了。


### 传感器类型各异

* 数据的类型还是变化太大

    图片和文本变不出花，但是传感器你单是加速度计和陀螺仪就可以完全不一样
* 结果

    单个数据集，不同领域在传感器各异的情况下，一个领域全是加速度计的数据，另一个全是陀螺仪的，迁移也是一个大问题。
## 迁移什么

可能从问题角度很难锁定HAR的迁移学习到底要干什么，但是一聊到HAR的迁移学习，还是千万不能思维定势，实际上目前可迁移的任务不少
### 单源领域的迁移

思维定势想到的迁移学习是第一种

__位于文章13页__

1. 同一套活动标签，同一套传感器，领域1迁移到领域2



### 多源领域迁移

单源领域迁移学习的变种

* 同一个数据集，同一套传感器类型，同一套活动标签
* 用多个源领域的数据，迁移到一个目标领域

    此时不再是串行输入数据到 __一个__ 模型，可能会有 __很多个__ 模型分别学习源领域



### 传感器多样性的迁移

这个特点只有少数的公开数据集具备

* 同一个数据集，同一套活动标签
* 每个领域可能使用不同类型的传感器收集数据

__作者在这块研究得很细致，论文第10页的4.1具体分成了三种，感兴趣直接去看__

1. m种传感器类型
2. p种传感器佩戴位置
3. d种同一类传感器的不同型号

    加速度计的不同型号

### 标签空间的迁移

第15页，稍后整理