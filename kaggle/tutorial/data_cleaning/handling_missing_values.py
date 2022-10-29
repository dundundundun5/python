import numpy as np
import pandas as pd
np.random.seed(0)
data_path = "../../input/titanic/train.csv"
data:pd.DataFrame = pd.read_csv(data_path)
# print(data.head())

# 获取每列的空缺数据
missing_values_count:pd.Series = data.isnull().sum()
print(missing_values_count)
# 通过df.shape获取行列数 df.shape[0]==rows df.shape[1]==cols
# 获取 缺失数据率 = 缺失数据格/总数据格
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing / total_cells) * 100
print(percent_missing)

# 1 dropna()，丢弃NAN行
print(data.dropna())
# dropna(axis=1)丢弃列
print(data.dropna(axis=1))

# 2 fillna(WORD) 默认填充所有NAN为WORD
print(data.fillna("aa"))
# fillna(method=)
# bfill、backfill 用下一个非缺失值填充该缺失值
# ffill、pad 用前一个非缺失值去填充该缺失值
print(data.fillna(method='bfill').fillna('aa'))