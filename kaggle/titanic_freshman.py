# copied from kaggle: titanic-tutorial
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk("./input/"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv("D:/temp_files/datasets/titanic/train.csv")
test_data = pd.read_csv("D:/temp_files/datasets/titanic/test.csv")

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./output/titanic/submission.csv', index=False)
print("Your submission was successfully saved!")