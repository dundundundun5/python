import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

data_path = "../../input/titanic/test.csv"
data = pd.read_csv(data_path)
print(data.tail())

# 柱状图
# plt.figure(figsize=(8, 8))
# plt.title("bar chart")
# x=横坐标数据，y=纵坐标数据
# sns.barplot(x=data.PassengerId, y=data.Age)
# plt.ylabel('Age of titanic')
# plt.xlabel('id of titanic')
# plt.show()
# sns.xxx(data=,x=,y=)xy可以指定横纵轴的数据
# 试着画Pclass种类的柱状图 y-数量 x-种类
# z = pd.DataFrame(data['Pclass'].value_counts())
# sns.barplot(x=z.Pclass, y=z.index)
# plt.ylabel('categories')
# plt.xlabel('amount')
# plt.show()


# 热力图
plt.figure(figsize=(5,5))
plt.title('heatmap of titanic')
# 显示相关系数矩阵 data=二维方阵 annot=是否在方格里显示数字
sns.heatmap(data=data.corr(), annot=True)
plt.show()