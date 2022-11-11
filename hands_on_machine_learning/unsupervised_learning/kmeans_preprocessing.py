from sklearn.datasets import load_digits
X_digits, y_digits =load_digits(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
# fit LR model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)

log_reg_score = log_reg.score(X_test, y_test)
print(log_reg_score)
# 创建一个流水线，使用Kmeans作为预处理
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
# 首先将训练集聚类为50个簇，并将图像替换为与这50个簇的距离
# 然后应用逻辑回归模型
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42))
])
pipeline.fit(X_train, y_train)
pipeline_score = pipeline.score(X_test, y_test)
print(pipeline_score)

# 用网格搜索选取k
from sklearn.model_selection import GridSearchCV
# 上面定义了流水线的第一步的名称为kmeans，这里kmeans__n_clusters,双下划线+参数名称就能改变算法超参？
# 这是什么用法，为什么从未见过？
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
# 查看网格搜索的最佳参数
print(grid_clf.best_params_)
print(grid_clf.score(X_test, y_test))