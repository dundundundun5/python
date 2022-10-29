import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy import stats
from mlxtend.preprocessing import minmax_scaling
data_path = "../../input/titanic/train.csv"
data:pd.DataFrame = pd.read_csv(data_path)
print(data.head())
# 获取某一浮点数列并转为numpy数组
fare_list = data['Fare'].to_numpy()
print(fare_list[:5])
# =======================================================================
# 对某一列归一化再拼装成dataframe
# using sklearn.prerocessing.MinMaxScaler
scaler = MinMaxScaler()
# MinMaxScaler要求有索引的dataframe，不能单单取出一个series
scaled_sklearn:np.ndarray = scaler.fit_transform(pd.DataFrame(data.index + data.Fare))
scaled_sklearn:pd.DataFrame = pd.DataFrame(scaled_sklearn, columns=['Fare'])
# using mlxtend.preprocessing.minmax_scaling
# 如果输入是series，返回会变成普通array，不希望用这种方法
# b = minmax_scaling(data.Fare, columns=[0])
# 如果输入df，返回也是df
scaled_mlxtend:pd.DataFrame = minmax_scaling(pd.DataFrame(data=data.index + data.Fare,columns=['Fare']), columns=['Fare'])
# print(a)
# print(b)
# ====================================================
# 对某一列数据转化为正态分布（高斯分布）
# 使用scipy.stats.boxcox()变换函数 数值必须严格大于0 所以会报错
index_of_positive_feature = data.Fare > 0
positive_feature = data.loc[index_of_positive_feature, 'Fare']
normalized_scipy = stats.boxcox(positive_feature)
normalized_scipy = pd.DataFrame(normalized_scipy[0], columns=['Fare'])
# using sklearn.prerocessing.StandardScaler
scaler = StandardScaler()
# 传入一个df，返回一个numpy array
normalized_sklearn:np.ndarray = scaler.fit_transform(pd.DataFrame(data.index + data.Fare))
# 重新封装
normalized_sklearn:pd.DataFrame = pd.DataFrame(normalized_sklearn,columns=['Fare'])
print(normalized_sklearn)
