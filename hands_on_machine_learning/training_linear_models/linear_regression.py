import numpy as np
# 随机生成Xy
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# np.c_按列拼接两个矩阵 np.ones(row, col)生成row行col列的1矩阵
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
# 标准方程
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# 用线性回归公式预测
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
# 用函数预测
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# 偏差项和特征权重
print(lin_reg.intercept_, lin_reg.coef_)
lin_reg.predict(X_new)
# ==================================
# 梯度下降算法快速实现
eta = 0.1 # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)
for iterations in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta * gradients
# 随机梯度下降快速实现
n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
# 选择自带随机梯度下降的SGDRegressor类
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None,
                       eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel()) # np.ravel()将y变为一维并返回y 函数要求的
print(sgd_reg.intercept_, sgd_reg.coef_)
# 小批量梯度下降（略）
# ==================================================
