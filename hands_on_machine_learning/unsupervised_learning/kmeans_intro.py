import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
blob_centres = np.array([
    [0.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3]
])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centres,
                      cluster_std=blob_std, random_state=7)
# import matplotlib.pyplot as plt
# matplotlib.use('tkagg')
# def plot_clusters(X, y=None):
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
#     plt.xlabel("$x_1$", fontsize=14)
#     plt.ylabel("$x_2$", fontsize=14, rotation=0)
# plt.figure(figsize=(8, 4))
# plot_clusters(X)
# plt.show()
# =======================
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k)
# 聚类并预测类别
y_pred = kmeans.fit_predict(X)
# labels_表示训练实例的类别副本
print(y_pred is kmeans.labels_)
# cluster_centres_为中心点坐标
print(kmeans.cluster_centers_)
# 创建靠近中心点的实例
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
print(kmeans.predict(X_new))
# 测量每个实例到每个中心点的距离, 最终会得到一个k维数据集，这是一种非线性降维技术
kmeans.transform(X_new)
# init=预设可能的中心点
# n_init=算法迭代次数 n_clusters=聚类数 默认使用kmeans++
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
print(kmeans.inertia_)

# 小批量kmeans不必在所有实例计算后才移动中心点
from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
print(minibatch_kmeans.inertia_)

# 轮廓系数
from sklearn.metrics import silhouette_score
#
print(silhouette_score(X, kmeans.labels_))
# 列表填充高级用法，高封装程度语言的优势
kmeans_per_k  = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

# 轮廓系数-k值的图
import matplotlib as mpl
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, 'bo-')
plt.xlabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
plt.show()
# k值=k'时，每个簇的轮廓系数
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")

    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

plt.show()
