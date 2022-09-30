import matplotlib
import scipy.sparse
import sklearn
import sys

assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# matplotlib.rc用于修改图的配置
# axes 坐标轴 xtick x轴刻度大小 y轴刻度大小
# labelsize 字体大小
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
# os.path.join()：连接两个或更多的路径名组件:
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs用于递归创建目录
# 如果 exist_ok 为 True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # 组合路径，补齐/
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        # plt.tight_layout()自动调整子图，使之填满整个图像区域
        plt.tight_layout()
    # plt.savefig(路径，format=文件格式字符串,dpi=分辨率)
    plt.savefig(path, format=fig_extension, dpi=resolution)


# =================================环境构造完毕================================================

import os
import tarfile
import urllib.request

# 下载数据集的源URL
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# 数据集的下载存放路径
HOUSING_PATH = os.path.join("datasets", "housing")
# 数据集的具体URL
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # 创建数据集应该坐落的文件夹
    os.makedirs(housing_path, exist_ok=True)
    # 设置压缩包应该坐落的路径
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # 参数1表示要复制到本地的网络对象的URL，参数2表示复制到本地的路径
    urllib.request.urlretrieve(housing_url, tgz_path)
    # 打开路径下的tar文件并返回tarfile对象
    housing_tgz = tarfile.open(tgz_path)
    # extractall()将压缩包的全部内容解压到path=指定路径
    housing_tgz.extractall(path=housing_path)
    # 关闭tarfile
    housing_tgz.close()


# 获取数据集并解压到指定路径
# fetch_housing_data()
# =================================下载数据集================================================
import pandas as pd


def load_housing_data(housing_path=HOUSING_PATH):
    # csv文件就是数据集
    csv_path = os.path.join(housing_path, "housing.csv")
    # 读取csv文件并转化为pandas.core.frame.DataFrame对象，该对象可以直接被print打印
    return pd.read_csv(csv_path)


housing = load_housing_data()  # type: pandas.core.frame.DataFrame
print(housing.head())  # 获取dataframe的头几(default=5)行
print(housing.info())  # 获取dataframe的简要摘要
# 获取dataframe某一列（这个数据集里列的索引是字符串）非空的行数
# 方便之处：重载运算符后，dataframe[str]可以获取列名=str的某一列
print(housing["ocean_proximity"].value_counts())
# 用于查看一些基本的统计详细信息，例如数据帧的百分位数，均值，标准差等或一系列数值
print(housing.describe())

# 绘制每个数值属性的直方图
print(type(housing))

# =================================读取数据集================================================
import matplotlib.pyplot as plt

# 使用TkAgg图形化渲染器，不写则无法显示
matplotlib.use("TkAgg")
# dataframe.hist()创建直方图
# figsize=创建图形的尺寸 bin=直方图的箱数
housing.hist(bins=50, figsize=(20, 15))
# 显示绘制的图片
plt.show()
# =================================数据集绘图================================================
# 创建测试集
import numpy as np


def split_train_test(data, test_ratio: float) -> [pd.DataFrame, pd.DataFrame]:
    """
    老旧的分割集合方法，每次运行都会创建不同的训练集:数据集，模型将逐渐接触到完整的数据集
    :param data: 待分割的数据集
    :param test_ratio: 测试集的占比
    :return: 训练集，数据集
    """
    # np.random.permutation(data)创建一个随机序列
    shuffled_indices = np.random.permutation(len(data))
    # 测试集长度
    test_set_size = int(len(data) * test_ratio)
    # 测试集选取特征向量的第一维索引值由随机序列的[0, 值为"测试集长度"的索引)决定
    test_indices = shuffled_indices[:test_set_size]
    # 训练集选取特征向量的第一维索引值由随机序列[值为"测试集长度"的索引, 末尾)
    train_indices = shuffled_indices[test_set_size:]
    # pd.iloc索引器用于按整数作为索引返回向量
    # 索引为单个数字则返回Series，索引为多个数字组成的列表则返回DataFrame对象
    return data.iloc[train_indices], data.iloc[test_indices]


# 创建训练集和测试集，实际为pd.DataFrame对象
train_set, test_set = split_train_test(housing, 0.2)  # type: pd.DataFrame
print(len(train_set))
print(len(test_set))

# =================================数据集分割方法1================================================
from zlib import crc32


def test_set_check(identifier, test_ratio):
    """
    校验函数，将n个n维特征向量的其中一个特征作为标识符，计算其循环冗余校验码(同理哈希地址法)

    如果小于校验码最大值(2的32次方，也就是机器码的数字表示的最大值)，则return

    :param identifier: 需要校验的标识符
    :param test_ratio: 测试集在数据集中的占比
    :return: True-校验码小于门槛 False-校验码大于门槛
    """
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio: float, id_column: str):
    """
    稳定的分割函数

    :param data: 数据集，m个n维向量组成的矩阵 m-数据个数 n-特征个数
    :param test_ratio: 测试集在数据集中的占比
    :param id_column: 将作为标识符进行校验的列名的键值（字符串）
    :return:
    """
    # 获取DataFrame.iloc[键值] 返回Series对象
    ids: pd.Series = data[id_column]
    # series.apply函数，参数是一个函数，对series的值进行批量处理
    in_test_set: pd.Series = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    # dataframe.loc[]输入
    # https://blog.csdn.net/gary101818/article/details/122454196
    # 1. 单个标签，例如5 or 'a'（请注意，它5被解释为索引的标签，而不是沿着索引的整数位置）。
    # 2. 标签列表或数组，例如.['a', 'b', 'c']
    # 3. 带有标签的切片对象，例如'a': 'f'。需要注意的是违背了普通的Python片，开始和停止都包括
    # 4. 与被切片的轴长度相同的布尔数组，例如.[True, False, True]
    # 5. 一个可对齐的布尔系列。键的索引将在屏蔽之前对齐。
    # 6. 一个可对齐的索引。返回选择的索引将作为输入。

    # 根据series序列的布尔值（参数的情况为4），取出布尔值为True的对应索引的向量
    return data.loc[~in_test_set], data.loc[in_test_set]


# =================================数据集分割方法2=============================================
# dataframe.reset_index() 方法将重置DataFrame表的索引，并使用默认索引
housing_with_id = housing.reset_index()
# 依据index进行集合划分
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# =================================按index进行集合划分=============================================
# 构建新的标识符列，id列的值=longitude列的值*1000+latitude列的值
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# 依据id进行集合划分
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
print(test_set.head())
# =================================按id进行集合划分=============================================
from sklearn.model_selection import train_test_split

# 调用库函数来分割集合而不是自行编写函数
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# =================================用sklearn.model_selection.train_test_split进行集合划分=============================================
# 实现分层抽样的分层，以报纸每一层分布均匀
# "income_cat"列为"median_income"列分割后标记的产物，
# 也就是用有限的数字对收入进行分类标签，分类的依据是收入中位数集中在1.5~6（万美元）
# 分为五个区间，区间的数字是依据数据分布人为设定的，五个区间的标签是1~5，最后该列每一行的值均为标签1~5
# housing["income_cat"]是一个series对象，其中每一行的元素是收入所属的label（1~5）
# strat_test_set["income_cat"] test_set["income_cat"]中的内容同理
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# 绘制直方图
housing["income_cat"].hist()
# 显示图片
plt.show()
# =================================分层抽样的分层=============================================
from sklearn.model_selection import StratifiedShuffleSplit

# StratifiedShuffleSplit()对数据集进行打乱划分
# n_splits=将训练数据分成train/test对的组数, test_size=测试集在数据集的占比, random_state=随机打乱种子
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# split() Generate indices to split data into training and test set.
# x - 训练数据 y- 基准数据 分层抽样将按照基准数据进行
# 看似是个循环，其实利用的是for 变量 in 集合的语法优势 调试过程中
for train_index, test_index in split.split(housing, housing["income_cat"]):  # type: np.ndarray
    # dataframe.loc[]相当于numpy array直接用[]表示某一行
    # 分层抽样的训练集
    strat_train_set = housing.loc[train_index]  # type: pd.DataFrame
    # 分层抽样的测试集
    strat_test_set = housing.loc[test_index]  # type: pd.DataFrame

# 查看测试集中收入类别的比例分布
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))


# =================================分层抽样的抽样=============================================
def income_cat_proportions(data):
    """
    查看某个集合中收入类别的比例分布
    :param data: 集合
    :return: 分布比例
    """
    # series.value_counts
    return data["income_cat"].value_counts() / len(data)


# 获取随机抽样的训练集、测试集
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# 组建pd.DataFrame，用于显示2种抽样的各类收入比例
# sort_index()返回按
compare_props: pd.DataFrame = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set)
}).sort_index()
# dataframe能直接通过[]来创建键值对
# 随机抽样的相对误差
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
# 分层抽样的相对误差（测量值/真值 * 100， 此处减去100是为了体现误差的正负，即测量相对真值略大还是略小）
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
# 打印dataframe
print(compare_props)
# =================================查看随机抽样和分层抽样中各收入类别在自己集合中的比例=============================================
# 抽样演示完了，所以"income_cat"列也就没用了，有用的留着没用的丢弃
# 遍历dataframe元组中的列
# dataframe.drop()方法通过指定标签名称和相应的轴，或直接给定索引或列名称来删除行或列
# axis=轴的方向，0为行，1为列，默认为0 inplace=布尔值，是否生效
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# 获取分层抽样后的训练集副本
housing = strat_train_set.copy()
# dataframe.plot
# kind=字符串类型，是固定的集合中的一个，一旦写错就报错 x=x轴标签 y=y轴标签
# s=散点图大小 label=, figsize=图片大小
housing.plot(kind="scatter", x="longitude", y="latitude")
# 显示图片
plt.show()
# alpha参数未查到
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
# colorbar=是否绘制颜色条 其余参数未知 pandas官网未查到
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
# =================================数据可视化入门=====================================
# 列的相关系数矩阵 numeric_only=Include only float, int or boolean data.
corr_matrix: pd.DataFrame = housing.corr(numeric_only=True)
# 按某一列的值降序排列dataframe中的行 ascending=升序与否
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)
# =================================检测数据相关性的方法1=====================================
from pandas.plotting import scatter_matrix

# 创建含有列名的列表
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# pandas.plotting.scatter_matrix将绘制 输入属性个数**2个图，用以展示列之间的相关性（此处4个属性总共是16个图）
scatter_matrix(housing[attributes], figsize=(12, 8))
# 显示图片
plt.show()
# 在16张途中发现了最富正相关特性的两列，故单独画出，alpha参数未知
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# 显示图片
plt.show()
# =================================检测数据相关性方法2=====================================
# 手动创建一些新的列，用于挖掘列之间相关性
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
# 新增3列以后再次获取相关系数矩阵
corr_matrix = housing.corr(numeric_only=True)
# 按房价中位数排序
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)
# =================================检测数据相关性的方法3=====================================
# "median_house_value"是标签，也就是需要预测的值，而承载机器学习算法的模型需要靠特征训练，所以要把标签和特征分开
# 从分层抽样的训练集中获取丢弃了标签所在列的余下dataframe axis=丢弃的是1/0 : 列/行
housing: pd.DataFrame = strat_train_set.drop("median_house_value", axis=1)
# 单独获取分层抽样训练集中的标签所在的列
housing_labels: pd.DataFrame = strat_train_set["median_house_value"].copy()
# dataframe.dropna()该函数是自解释的，丢弃指定列中值为NAN的行
# subset=column label or sequence of labels, optional
# housing.dropna(subset=["total_bedrooms"])  # option 1
# # 更加暴力的方法，丢弃整个列，而不去处理少数个的NAN
# housing.drop("total_bedrooms", axis=1)  # option 2
# # 中位数填充法，获取该列的值的中位数，其实和删除整列差别不大，但是对数据清洗是有贡献的
# median = housing["total_bedrooms"].median()  # option 3
# # dataframe.fillna()自解释的方法，对某一个列的所有NAN填充某个数字 inplace=是否就地生效
# housing["total_bedrooms"].fillna(median, inplace=True)

# =================================异常值的统一处理方法=====================================
from sklearn.impute import SimpleImputer

# sklearn.impute.SimpleImputer
# strategy=空值填充策略mean、median、most_frequent、constant
imputer = SimpleImputer(strategy="median")
# 因为文本没有中位数，故先获取只有数字的dataframe
housing_num = housing.drop("ocean_proximity", axis=1)
# SimpleImputer.fit()可以计算矩阵缺失的相关值的大小（统计时将排除某列的nan行），以便填充其他缺失数据矩阵时进行使用。
imputer.fit(housing_num)
# 查看每个数字列的中位数，检查其是否与SimpleImputer类计算中位数保持一致
print(housing_num.median().values)
# 显示SimpleImputer.statistics_，表示计算好的相关值（相关值向量）
print(imputer.statistics_)
# 计算完毕后，对数字列的nan进行填充，此处赋值给X是因为转换后需要用变量接受返回值
# housing_num目前还是存在nan的，而X是一个np.ndarray，是一个二维数组
# 丢失了列的名称但某些列种原本为nan的行已被相关值填充
X: np.ndarray = imputer.transform(housing_num)
# 由于返回值是np.ndarray还需把填充好后的数字列转换为dataframe
housing_tr: pd.DataFrame = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
# =================================数字缺失值的处理方法（sklearn自带）=====================================
# 获取文本列，此处传入列表或字符串
# 传入字符串返回值是series，传入列表（即使列表里只有一个字符串）返回值是dataframe
housing_cat: pd.Series = housing[["ocean_proximity"]]
print(housing_cat.head(10))
# 导入sklearn.preprocessing.OrdinalEncode 用于普通的分类编码
# 但是各编码之间是存在距离远近（而不是距离一致相等）
from sklearn.preprocessing import OrdinalEncoder

# 新建编码对象
ordinal_encoder: OrdinalEncoder = OrdinalEncoder()
# 获取编码后的数组 np.ndarray
housing_cat_encoded: np.ndarray = ordinal_encoder.fit_transform(housing_cat)
# 类型为数组，所以只能利用np.ndarray[column_st:column_ed, row_st:row_ed]语法来获取行
print(housing_cat_encoded[:10])
# 被编码的某一列有多少种类别的字符串
print(ordinal_encoder.categories_)
# =================================文本属性值的编码方法1=====================================
# 独热编码简介 : [认知男生理男,认知男生理女,认知女生理女,认知女生理男]
# 独热编码结果: [0001, 0010, 0100, 1000] 类别 = 位数
from sklearn.preprocessing import OneHotEncoder

# 独热编码类对象创建
cat_encoder: OneHotEncoder = OneHotEncoder()
# 稀疏数组：数字位置的统计方法改变，
# 不再是按每个单元格个统计，而是只统计值有效的单元格
# 类型是稀疏矩阵scipy.sparse.csr_matrix
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# 稀疏矩阵转化为数组 np.ndarray
print(housing_cat_1hot.toarray())
# =================================文本属性值的编码方法2=====================================
from sklearn.base import BaseEstimator, TransformerMixin

# 下面定义一个自动转换器，自动根据现有列生成一些额外列，额外列是现有列之间运算得到的结果
# option 1 因为在dataframe-housing变量的可视化显示里，列号是从0开始计数的
# total_rooms,total_bedrooms, population, households分别在3、4、5、6列
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
# option 2
col_names = ["total_rooms", "total_bedrooms", "population", "households"]


def get_column(data: list) -> list:
    res = []
    for c in col_names:
        res.append(housing.columns.get_loc(c))
    return res


[rooms_ix, bedrooms_ix, population_ix, households_ix] = get_column(col_names)  # type: int

# option 3 for的高级用法 目前还没学会
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names
]


# 转换器类 继承TransformerMixin类是为了覆盖fit_transform()方法
# 不继承也毫无影响
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        """
        转换类构造函数
        :param add_bedrooms_per_room: 是否添加列名="每套房卧室数量"
        """
        # 这是一个临时定义的对象属性的（局部？）变量，用于告知transform函数是否要加上列名="每套房卧室数量"
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """
        没有任何意义的函数
        :param X:
        :param y:
        :return:
        """
        return self  # nothing else to do

    def transform(self, X: np.ndarray):
        room_per_household: np.ndarray = X[:, rooms_ix] / X[:, households_ix]
        population_per_household: np.ndarray = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room: np.ndarray = X[:, rooms_ix] / X[:, households_ix]
            # np.c_[matrix1, matrix2, ...] 要求行数相等，按列叠加多个矩阵（列向量也算）
            return np.c_[X, room_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, room_per_household, population_per_household]


"""
Also, `housing_extra_attribs` is a NumPy array,
 we've lost the column names
(unfortunately, that's a problem with Scikit-Learn). 
To recover a `DataFrame`, you could run this:
"""

# 把原来的DataFrame转换（利用转换器）
attr_addr = CombinedAttributesAdder(add_bedrooms_per_room=False)
# 返回的是numpy.ndarray，从而失去了pandas.DataFrame的属性名
housing_extra_attribs = attr_addr.transform(housing.values)

# print(type(housing_extra_attribs))
# 因为np.c_[]返回的是np.ndarray，将原本dataframe的列名模糊成了数字索引，故
# 需要重新封装为pandas.DataFrame

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs, # data=ndarray (structured or homogeneous), Iterable, dict, or DataFrame
    # columns=[column_name1, column_name2, ...]
    columns=list(housing.columns) + ["room_per_household", "population_per_household"],
    # index=Index or array-like
    index=housing.index
)
print(housing_extra_attribs.head(20))
# =================================自定义转换器=====================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 数字转换流水线类
# 会根据列表内的元组，对所有数字列进行流水线级转换，
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])
# 拟合并转换
housing_num_tr:pd.DataFrame = num_pipeline.fit_transform(housing_num)
# =================================使用sklearn的数字转换流水线类=====================================
from sklearn.compose import ColumnTransformer

# 能够处理所有列的转换器
# 单独获取数字列的列名称组成的列表（对dataframe进行list类型转换能获取？）
num_attribs: list = list(housing_num)
# 单独获取文本列的列名称组成的列表
cat_attribs = ["ocean_proximity"]
# 对数字列和文本列分别是施加流水线
# 格式为 ColumnTransformer(元组列表)
# 元组列表=[("任意名称",转换器,转换列的列名组成的列表), (...), (...), ...]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
# 使用统一流水线函数对所有列进行转换
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
# =================================使用sklearn的统一转换流水线类=================================
# 训练线性回归模型
from sklearn.linear_model import LinearRegression
# 创建一个空白的线性回归模型
lin_reg = LinearRegression()
# fit(统一转换流水线处理后的特征向量们，训练集的的标签（房价中位数）) 对模型进行拟合
lin_reg.fit(housing_prepared, housing_labels)
# 从训练集种拿出部分数据
some_data: pd.DataFrame = housing.iloc[:5]
some_labels: pd.DataFrame = housing_labels.iloc[:5]
# 用统一转换流水线处理部分数据
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# 衡量模型的RMSE均方根误差
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
# 均方误差
lin_mse = mean_squared_error(housing_labels, housing_predictions)
# 均方根误差
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# 直接获取RMSE
# lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
# =================================线性回归模型的拟合、预测、评估=================================
# 获取平均绝对误差
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
# print(lin_mae)

# 选择决策树模型
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# 训练集评估决策树模型
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# 0.0?  是否是过拟合了
# print(tree_rmse)


from sklearn.model_selection import cross_val_score

# 利用10折交叉验证产生一个包含10次评估分数的数组
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
# sklearn生成负的MSE函数
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standart deviation:", scores.std())


# display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)

# 用随机森林模型
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
# print(forest_rmse)
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

# 用gridsearch网格搜索微调模型
from sklearn.model_selection import GridSearchCV

# 首先评估 3 * 4共12种超参数组合
# 然后评估 2 * 3共6种超参数组合 bootstrap = False(default:True)
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)
# print("开始")
grid_search.fit(housing_prepared, housing_labels)
# # print(grid_search.best_params_)
# # print(grid_search.best_estimator_)
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
# print(pd.DataFrame(grid_search.cv_results_))

# RandomizedSearch随机搜索微调模型
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
# param_distribs = {
#     "n_estimators": randint(low=1, high=200),
#     "max_features": randint(low=1, high=8),
# }
# forest_reg = RandomForestRegressor(random_state=42)
# rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                 n_iter=10, cv=5, scoring="neg_mean_squared_error", random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)
# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

# 指出每个属性的相对重要程度
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
extra_attribs = ["room_per_hhold", "pop_perr_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
# 使用scipy.stats.t.interval()计算泛化误差的95%置信区间
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
g = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))
print(g)
