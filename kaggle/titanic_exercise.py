import pandas as pd

train_path = "./input/titanic/train.csv"
test_path = "./input/titanic/test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# print(train_data.info())
# 使用平均年龄来填充年龄中nan值
train_data['Age'].fillna(train_data.Age.mean(),inplace=True)
test_data['Age'].fillna(test_data.Age.mean(),inplace=True)

# 查找缺失值
# print(train_data[pd.notnull(train_data.Embarked)])
# 观察Embarked值的种类
# print(test_data.Embarked.value_counts())
# S最多
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
# Cabin 暂时还不知道怎么处理，先从特征中删掉
# 查看keys
# print(test_data.keys())
# 特征提取
# TODO learn to implement DictVectorizer



# # 特征选择
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# X = train_data.loc[:, features]
# y = train_data['Survived']
# from sklearn.model_selection import train_test_split
# train_X, validate_X,train_y, validate_y= train_test_split(X, y, random_state=222200801)
# test_X = test_data.loc[:, features]
#
#
# from sklearn.tree import DecisionTreeClassifier
# tree_model = DecisionTreeClassifier(random_state=1,max_depth=None,max_leaf_nodes=None,criterion='gini')


