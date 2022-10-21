import pandas as pd
data = pd.read_csv("../../input/titanic/test.csv")
print(data.head())


# df.attribute 通过类C结构体的方式获取列
print(data.PassengerId)
# 同样，可以通过df['attribute']获取列
print(data['PassengerId'])
# 对于筛选出的列，还可以获取列中的某行
print(data['PassengerId'][416])

# iloc[IDX] 按索引筛选出行
print(data.iloc[417])
# iloc[row,column]按行列筛选数据
# row 可以是int,list[int], int=st:int=ed->[st,ed)
# 其中st默认为0，ed默认为最大索引+1
# 筛选第0和1列，22行数据（左开右闭）
print(data.iloc[22,0:2])
# 筛选第一行开始的0和1列的数据
print(data.iloc[1:, [0, 1]])
# 筛选倒数第5、4、3、2行的所有数据
print(data.iloc[-5:-1])

# loc[row, column]可用行列名获取df的行列,
# 而通常不能用数字作为索引（如果行索引不是数字）
# 如果行索引是数字，则不存在左闭右开的规则
# 获取行索引为0、1、2、3的Pclass列
print(data.loc[[0, 1, 2, 3], 'Pclass'])
# 通过列名的列表获取多个列
print(data.loc[:, ['Name', "PassengerId"]])

# df.set_index()将原有索引替换为现有的某个标签
# 将PassengerId作为索引
print(data.set_index("PassengerId"))
# df.reset_index()将对索引重新编号
print(data.reset_index())
# loc条件选择，选择Pclass为1的行
print(data.loc[data.Pclass==1])
# 条件选择的多条件合并 选择Pclass为1且Embarked为S的行
print(data.loc[(data.Pclass==1)&(data.Embarked=="S")])
# 条件选择除了使用python自带逻辑判断，还可以使用isin()
# data.attribute.isin([range])
print(data.loc[data.PassengerId.isin([906, 912])])
# isnull notnull可以筛选出为空、不空的行
print(data.loc[data.Cabin.notnull()])

# 给某一列统一赋值
data_temp = data
data_temp['Cabin'] = "CAONIMA"
print(data_temp.head())
# 用range给列赋值, 对index反向赋值
id_temp = data_temp.loc[:,'PassengerId']
# 要表示1~25 python中为[1,25+1)=[1,26)步进=1,
# 如果要反向，则为[25, 1-1)=[25,0)
data_temp['PassengerId'] = range(id_temp[len(data) - 1], id_temp[0] - 1,-1)
# 此时最后一行PassengerId为892
print(data_temp.iloc[[-2,-1,0],:])