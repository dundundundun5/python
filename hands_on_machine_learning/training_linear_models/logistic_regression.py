from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
# print(iris.DESCR)
print(iris.keys())
X = iris["data"][:, 3:]  # pedal width
y = (iris['target'] == 2).astype(np.int64) # 1 if iris virginica, else 0

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X,y)
# P137
print(log_reg.predict([[1.7], [1.5]]))