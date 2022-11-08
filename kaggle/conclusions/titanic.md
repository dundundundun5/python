# Titanic - Machine Learning from Disaster
page-><https://www.kaggle.com/competitions/titanic>
施工中······
# 预处理
由于数据集已经分为测试集和训练集，故读取后分别进行预处理

训练集随后又被划分为验证集和小训练集，目前没有作用


## 缺失值
* Age年龄

    均值填充
* Fare费用

    均值填充
* Embarked出发港口

    使用dataframe.value_counts()后发现'S'为最多，故用S填充
* Cabin舱号

    缺失值数量极多，对该特征暂时性放弃
## 特征提取
* 提取方法

    pd.get_dummies(pd.Series)返回一个DataFrame，将要提取的特征提取出来，从而可以把提取好的特征按列拼接在原DataFrame后(使用pd.concat([df1,df2,...,dfn], axis=1))，此时未处理的df和提取了特征的df已经合并，因而要把原来未提取的特征丢弃
## 特征缩放
* 归一化

    原理如下：
    $$x_{scaled} = \frac{x_{raw} - \min{X}}{\max{X}-\min{X}} $$
    $$x_{scaled}\longrightarrow缩放后的数据$$
    $$x_{raw}\longrightarrow缩放前的数据$$
    $$\min{X}\longrightarrow数据的最小值$$
    $$\max{X}\longrightarrow数据的最大值$$
    编程实现方法:

    ```
    1. 创建scaler = sklearn.preprocessing.MinMaxScaler
    2. scaler.转换函数(pd.DataFrame)->np.ndarray
    3. 将np.ndarray重新封装为pd.DataFrame
    4. 按列拼接随后丢弃未被归一化的原始列
    ```

## 特征选择
抛弃了可排序但没有实际意义的列（PassengerId等）
# Decision Tree 
使用sklearn.tree.DecisionTreeClassifier

# MLP 
使用tensorflow.keras.Dense

# 这个题目在未来的目标
1. 尝试手写mlp（初学不太可能），退一步是阅读成品api的mlp源码
2. 用随机搜索寻找、调整超参数，让验证集派上用途