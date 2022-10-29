import matplotlib
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

data_path = "../../input/titanic/train.csv"
data = pd.read_csv(data_path)
print(data.tail())


# 调整图像(窗口)的宽和高 单位inch
plt.figure(figsize=(6,6))
# lineplot(data=)折线图 还有barplot heatmap
sns.lineplot(data=data)
plt.show()
# 画单个列的图
# 设置标题
plt.title("titanic subset")
# 画单个列的折线图， label=折现标签
sns.lineplot(data=data['Age'], label='Age of training set')
sns.lineplot(data=data['Pclass'], label='Pclass of XXXXXX')
# 横坐标名称设置
plt.xlabel('PassengerId')
plt.show()