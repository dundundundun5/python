import pandas as pd
data = pd.read_csv("../../input/titanic/train.csv")
# rename()可对标签重命名
print(data.rename(columns={'Pclass': '哈哈'}))
# 还可对索引重命名（罕见）
print(data.rename(index={0:'无坚不摧'}))
# 还可以对行列起名
print(data.rename_axis('idx', axis='rows').rename_axis('info', axis='columns'))

data_test = pd.read_csv("../../input/titanic/test.csv")
# pd.concat()把两个数据集拼在一起，要求行列一致
data_concated = pd.concat([data, data_test])
# 用赋值运算符重设index值
# data_concated.index = data_concated.reset_index().index
print(data_concated)


# join可以合并两个数据集，按一或多个标签索引后，可以合并查看两个数据集的对照信息
# 设置双标签（类似groupby的双标签分组）
left = data.set_index(['Sex','Pclass'])
right = data_test.set_index(['Sex','Pclass'])
# lsuffix rsuffix是防止出现对照是列标签重复
# 查看按性别、pclass分类的数据中，各种信息的对照表
joined = left.join(right, lsuffix='_train', rsuffix='_test')
print(joined)
