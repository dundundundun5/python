import pandas as pd
data = pd.read_csv("../../input/titanic/train.csv")
print(data)
# groupby(A) 按A列分组
# 如果A的值为1、2、3
# 分别A=1 A=2 A=3时有多少示例数
print(data.groupby('Pclass').Pclass.count())
# 按某列分组后，取某个其他属性的统计值
# 按Pclass分组后分别统计年龄的最大值
print(data.groupby('Pclass').Age.max())

# 分组后得到的子df也可以施加统计方法
# 按Pclass分组后，取每个分组的Cabin列，列出0、1、2行
print(data.groupby('Pclass').apply(lambda df: df.Cabin.iloc[0:3]))
# 还可以按多标签分类
# 按Sex、Pclass分组后，对每个分组取出PassengerId最大的一行中的两列
# PassengerId和Fare列
print(data.groupby(['Sex', 'Pclass']).apply(lambda df: df.loc[df.PassengerId.idxmax(),
                                                              ['PassengerId', 'Fare']]))
# agg([算子])可对分组同时计算函数值
print(data.groupby('Pclass').Age.agg([len, min, max]))

# 多标签
data_grouped = data.groupby(['Sex', 'Pclass']).Age.agg([max, min])
mi = data_grouped.index
print(type(mi))
data_grouped = data_grouped.reset_index()
print(data_grouped.head())

# 数据排序 by:标签 ascending：True升序False降序
data_sorted = data.sort_values(by='Fare', ascending=False)
print(data_sorted.head())
# 按索引排序
print(data.sort_index().head())
# 数据排序多排序，传入列表即可
data_sorted = data.sort_values(by=['Fare', 'PassengerId'], ascending=False)
print(data_sorted.head())