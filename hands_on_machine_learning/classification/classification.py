import matplotlib

import sys
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab or Kaggle?
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
# ========================================================
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

print(mnist.keys())
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()
# ==================================
y = y.astype(np.uint8)

# 画图可视化，暂不考虑
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.show(image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size, images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap=mpl.cm.binary, **options)
    plt.axis("off")

# =========================================================
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# 随机梯度下降分类器
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
# ===============================================
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# 使用分层抽样在训练集中手动分割并交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
# 分层抽样后的折叠集合
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# 手动交叉验证
for train_index, test_index in skfolds.split(X_train, y_train_5):

    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
# 使用cross_val_score()验证
from sklearn.base import BaseEstimator
# 创建一个结果为全非的分类器，设定为准确率
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
never_5_clf = Never5Classifier()
# 该分类器准确率很高，但明显没有泛化能力
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# =================================
# 混淆矩阵评估分类器
# 使用k折交叉预测获取训练集的预测，从而计算训练集的混淆矩阵
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# 混淆矩阵
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred))
# 假装全对了
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)
# ======================================
# 召回率（敏感度）：实际为正实例中预测为正的比率 阳性检测出阳的概率
# 精度： 预测为正实例中实际为正的比率
from sklearn.metrics import precision_score, recall_score
# 函数获取精度\召回率
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
# F1分数 =... p93 是精度召回率的谐波平均值
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train)
# 混淆矩阵验证精度召回率f1
cm = confusion_matrix(y_train_5, y_train_pred)
cm[1, 1] / (cm[0, 1] + cm[1, 1])
cm[1, 1] / (cm[1, 0] + cm[1, 1])
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)
# ==============================
# 精度召回率权衡：决策边界的调整 P93
#  decision_function返回实例的分数，
y_scores = sgd_clf.decision_function([some_digit])
# sgdclf默认阈值为0，阈值设置为与默认一致
threshold = 0
y_some_digit_pred = (y_scores > threshold)
# 再设置
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
# 获取训练集所有实例的分数，来决定使用什么阈值
# 返回的是实例分数而不是预测见过
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3
                             ,method='decision_function')
# 计算所有可能的阈值的精度召回率
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# 作图代码省略
# 获取90精度的门限
# np.argmax()返回列表中最大值的索引值
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90= (y_scores >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)
# ====================================
# ROC曲线 召回率和fpr关系
# fpr + tnr = 1
# fpr实际为假中被预测为真的比例 误报率 阴性查出阳性的概率
# tnr特异度预测为假中实际为假的比例 阴性检测出阴性的概率
# auc线下面积越大越好 最大为1
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# 作图代码神略
# 获取roc线下面积
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
# =======================================
# 多分类器 p100
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train, not y_train_5
svm_clf.predict([some_digit])
some_digit_scores = svm_clf.decision_function([some_digit])
