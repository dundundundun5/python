import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


## 特征提取

features_need_dummied: list[str] = ['Embarked','Sex']


def implement_dummies(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    res = pd.DataFrame()
    for feature in features:
        dummy = pd.get_dummies(data[feature])
        res = pd.concat([res, dummy], axis=1)
    data = data.drop(labels=features, axis=1)
    data = pd.concat([data, res], axis=1)
    return data


X = implement_dummies(X, features_need_dummied)
test_X = implement_dummies(test_X, features_need_dummied)


# 归一化
def implement_minmax_scaler(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    res = pd.DataFrame()
    for feature in features:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pd.DataFrame(data.index + data[feature]))
        scaled = pd.DataFrame(scaled, columns=[feature])
        res = pd.concat([res, scaled], axis=1)
    data = data.drop(labels=features, axis=1)
    data = pd.concat([data, res], axis=1)
    return data


features_need_scaled = ['Fare', 'Age']

X = implement_minmax_scaler(X, features_need_scaled)
test_X = implement_minmax_scaler(test_X, features_need_scaled)
a = X.shape
from sklearn.model_selection import train_test_split
X, valid_X, y, valid_y = train_test_split(X, y)

from tensorflow import keras
mlp_model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[X.shape[1]]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
mlp_model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history = mlp_model.fit(X, y, epochs=30,validation_data=(valid_X, valid_y))
import numpy as np
predictions = np.argmax(mlp_model.predict(test_X), axis=1)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./output/titanic/submission_mlp.csv', index=False)
print("Your submission was successfully saved!")