import pandas as pd
data = pd.read_csv("../../input/titanic/train.csv")
# df.attribute.dtype 可获取属性的数值类型
# df.dtypes 可获取数值类型统计
# str类型通常为object
print(data.dtypes)
print(data.index.dtype)

# astype()数据类型转换
# PassengerId int64->float64
print(data.PassengerId.astype('float64'))

# 找出data.Cabin=NAN的行
print(data[pd.isnull(data.Cabin)])
# 多种找NAN
# missing_price_reviews = reviews[reviews.price.isnull()]
# n_missing_prices = len(missing_price_reviews)
# # Cute alternative solution: if we sum a boolean series, True is treated as 1 and False as 0
# n_missing_prices = reviews.price.isnull().sum()
# # or equivalently:
# n_missing_prices = pd.isnull(reviews.price).sum()
# 填充data.Cabin=NAN
data_temp = data.Cabin.fillna("无")
print(data_temp.head())
# 替换其他值
# 注意到data_temp是series，而不是df
data_temp = data_temp.replace("无","abc")
print(data_temp.head())
# 要替换data，必须对某个列series进行覆盖赋值
data.loc[:,'Cabin'] = data.Cabin.fillna("NONE")
print(data.head())