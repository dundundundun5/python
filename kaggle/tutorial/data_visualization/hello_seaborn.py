import pandas as pd
import matplotlib as mlt
mlt.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "../../input/titanic/train.csv"
data = pd.read_csv(data_path)
print(data.head())
# 设置图片长宽
plt.figure(figsize=(16, 6))
sns.lineplot(data=data)
plt.show()