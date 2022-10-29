import pandas as pd
import numpy as np
train_path = "./input/titanic/train.csv"
test_path = "./input/titanic/test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print(train_data.info())
print(test_data.info())

"""
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""
# print(train_data.info())
# 先把小于1的
train_data.loc[train_data.Age < 1, 'Age'] = 1
test_data.loc[test_data.Age < 1, 'Age'] = 1
# 使用平均年龄来填充年龄中nan值
train_data['Age'].fillna(train_data.Age.mean(), inplace=True)
test_data['Age'].fillna(test_data.Age.mean(), inplace=True)
test_data['Fare'].fillna(test_data.Fare.mean(), inplace=True)
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

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']
# from sklearn.model_selection import train_test_split
# train_X, validate_X,train_y, validate_y= train_test_split(X, y, random_state=222200801)
test_X = test_data.loc[:, features]


# 性别转bool
X.loc[:, 'Sex'] = (X.Sex == 'male').astype(np.int64)
test_X.loc[:, 'Sex'] = (test_X.Sex == 'male').astype(np.int64)
## 特征提取

features_need_dummied:list[str] = ['Embarked']
def implement_dummies(data:pd.DataFrame, features:list[str])->pd.DataFrame:
    res = pd.DataFrame()
    for feature in features:
        dummy = pd.get_dummies(data[feature])
        res = pd.concat([res, dummy], axis=1)
    data.drop(labels=features, inplace=True, axis=1)
    data = pd.concat([data, res], axis=1)
    return data
X = implement_dummies(X, features_need_dummied)
test_X = implement_dummies(test_X, features_need_dummied)
# 归一化
def implement_minmax_scaler(data:pd.DataFrame, features:list[str])->pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    res = pd.DataFrame()
    for feature in features:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pd.DataFrame(data.index + data[feature]))
        scaled = pd.DataFrame(scaled, columns=[feature])
        res = pd.concat([res, scaled], axis=1)
    data.drop(labels=features, axis=1, inplace=True)
    data = pd.concat([data, res], axis=1)
    return data

features_need_scaled = ['Fare', 'Age']

X = implement_minmax_scaler(X, features_need_scaled)
# 训练一个简单的决策树分类器
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=222)
tree_model.fit(X, y)
test_y = tree_model.predict(test_X)

# 网格搜索、随机搜索

submission = pd.DataFrame({'PassengerId': test_X.PassengerId, 'Survived': test_y})
submission.to_csv('./output/titanic/submission.csv', index=False)

