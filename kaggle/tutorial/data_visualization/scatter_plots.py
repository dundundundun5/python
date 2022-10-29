import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

data_path = "../../input/titanic/test.csv"
data = pd.read_csv(data_path)
print(data.tail())

# 回归拟合曲线
sns.regplot(x=data['Age'], y=data['Fare'])
# 散点图 xy含义不变，hue-用颜色区分第三类
sns.scatterplot(x=data['Age'], y=data['Fare'], hue=data.Pclass,color=['r','g','b'])



plt.show()