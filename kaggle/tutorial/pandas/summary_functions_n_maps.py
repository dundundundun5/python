import pandas as pd
data = pd.read_csv("../../input/titanic/train.csv")

# 查看统计总览
print(data.describe())
# 查看某一列的统计
print(data.Pclass.describe())
# 查看平均值
print(data.Age.mean())
# 查看值的种类，返回列表
print(data.Cabin.unique())
# 查看值的每个种类的出现次数
# 返回pd.series
print(data.Pclass.value_counts())
# idxmax()将统计series中的的最大值的行索引
age_max = data.Age.idxmax()
print(age_max)
# map(function())计算现有列的值的线性组合，返回series
age_mean = data.Age.mean()
age_new = data.Age.map(lambda a: a - age_mean)
print(age_new)

# apply()和map()功能一样 但是返回df
def remean_age(row):
    row.Age = row.Age - age_mean
    return row
data_temp = data.apply(remean_age, axis='columns')
print(data_temp.loc[:,'Age'])

# 如果追求速度也可以只作列的运算而不使用函数
print(data.Age - age_mean + 100)
print(data.Cabin + " haha")