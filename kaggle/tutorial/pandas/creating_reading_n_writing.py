import pandas
import pandas as pd
# dataframe输入为字典，每个键值名对应列名，后面的列表对应该列所有示例
a = pd.DataFrame({"Yes": [50, 21], "No": [131, 2]})
# dataframe列的数据类型不限
a = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
# dataframe还可以指定索引名
a = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']},
                 index=["product a", "product b"])
# series是dataframe的一列
a = pd.Series([1, 2, 3, 4, 5])
# series没有列名，只有一个统一的名字name
a = pd.Series([1, 2, 3, 4, 5], name="Product A")

# pd.read_csv(filepath)读取数据文件
data = pd.read_csv("../../input/titanic/test.csv")
# df.shape返回（行，列）
print(data.shape)
# df.head()默认返回头5行数据
print(data.head())
# index_col参数去除数据原有的索引
data = pd.read_csv("../../input/titanic/test.csv", index_col=0)
print(data.head())
# df.to_csv(filepath)将df存储到硬盘
