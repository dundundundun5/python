import numpy as np
from functools import partial
from numpy.linalg import norm


def golden_section_search(low, high, f, epoch=5, LAMBDA=0.618):
    # 左右点的初值
    left = low + (1 - LAMBDA) * (high - low)
    right = low + LAMBDA * (high - low)
    for i in range(epoch):
        # 选取计算出在数轴左边的点和右边的点
        left, right = (right, left) if left > right else (left, right)
        # print("区间为:[%.3f,%.3f]" % (low, high))
        # print(f"选取点为left={left},right={right}")
        f_left, f_right = f(left), f(right)
        if f_left > f_right:
            low = left
            # 左点变为区间端点，右点变为保留点，计算新左点
            left = low + (1 - LAMBDA) * (high - low)
        else:
            high = right
            # 同理
            right = low + LAMBDA * (high - low)
    return (low + high) / 2


def function(x: np.ndarray, A, b, c):
    """

    :return: 函数值
    """
    res = 1 / 2 * np.matmul(x.T, np.matmul(A, x)) + b.dot(x) + c
    return res


def gradient(x: np.ndarray, A, b, c):
    """

    :param x: 初始点
    :param A: 二次型
    :param b: 一次向量
    :param c: 常数
    :return: 二次型梯度
    """
    return np.matmul(A, x) + b


def get_function_l(l, x, d, A, b, c):
    """

    :param l: lambda
    :param d: 该点负梯度方向
    :return: 使f(x)转化为f(lambda) 返回 f(lambda)
    """
    new_x = []
    for i in range(len(x)):
        new_x.append(x[i] + d[i] * l)
    new_x = np.array(new_x)
    return function(new_x, A, b, c)


def optimal_gradient_search(x, A, b=np.array([0, 0]), c=0, epsilon=0.02, epochs=5):
    for epoch in range(epochs):
        # 获取该点负梯度
        d = -gradient(x, A, b, c)
        if np.linalg.norm(d) <= epsilon:
            print("最小值x*=", x)
            break
        # 对于f,固定x/d/A，使得lambda为变量
        f = partial(get_function_l, x=x, d=d, A=A, b=b, c=c)
        res_l = golden_section_search(0, 1, f)  # 求出最优的lambda

        # 对x向量的每个元素进行迭代
        for i in range(len(x)):
            x[i] = x[i] + res_l * d[i]


def conjugate_gradient_search(x: np.ndarray, A, b=np.array([0, 0]), c=0, epochs=2):
    d: np.ndarray = -gradient(x, A, b, c)
    for epoch in range(epochs):
        # 按公式计算lambda
        l = - (gradient(x, A, b, c).dot(d)) / (d.dot(np.matmul(A, d)))
        x = x + d * l
        # 按公式计算beta
        beta = (gradient(x, A, b, c).dot(np.matmul(A, d))) / (d.dot(np.matmul(A, d)))
        # 找下一个共轭方向
        d = -gradient(x, A, b, c) + beta * d
    print("最小值x*=", x)


def fletcher_revees_conjugate_gradient_search(x: np.ndarray, A, b=np.array([0, 0]), c=0, epochs=10):
    # 初始为负梯度方向
    d: np.ndarray = -gradient(x, A, b, c)
    for epoch in range(epochs):
        f = partial(get_function_l, x=x, d=d, A=A, b=b, c=c)
        res_l = golden_section_search(-10, 10, f)  # 求出最优的lambda
        pre = x # 保存上一个x
        x = x + res_l * d
        # 公式计算下一个共轭方向
        d = -gradient(x, A, b, c) + \
            (pow(norm(gradient(x, A, b, c)), 2)) / (pow(norm(gradient(pre, A, b, c)), 2)) * d
    print("最小值x*=", x)


def powell_search(x: np.ndarray, D, A, b=np.array([0, 0]), c=0, epochs=2):
    x_0 = x # 保存初始点方向
    for i in range(epochs + 1):
        # 最后一轮的共轭方向用x_n-x_0替代
        d = D[i] if i is not epochs else x - x_0
        f = partial(get_function_l, x=x, d=d, A=A, b=b, c=c)
        res_l = golden_section_search(-20, 20, f)  # 求出最优的lambda
        x = x + res_l * d
    print("最小值x*=", x)

def variable_metric_search(x: np.ndarray, H:np.ndarray, A:np.ndarray, b=np.array([0, 0]), c=0, epochs=2):
    # python求hessian矩阵比较困难，没做出来，故省略
    pass

if __name__ == '__main__':
    # 最小值=[0,0]
    optimal_gradient_search(x=np.array([1.0, 1.0]), A=np.array([[1,0],[0,2]]), b=np.array([0,0]), c=0)
    # 最小值=[0,0]
    conjugate_gradient_search(x=np.array([10., -5.]), A=np.array([[1, 1], [1, 2]]), b=np.array([0, 0]), c=0)
    # 最小值=[0,0]
    fletcher_revees_conjugate_gradient_search(x=np.array([10., -5.]),A=np.array([[1,1],[1,2]]), b=np.array([0,0]), c=0)
    # 最小值=[10.48,-0.92]
    powell_search(x=np.array([20., 20.]), D=np.array([[1, -1], [1, 1]]), A=np.array([[2, 0], [0, 4]]), b=np.array([0, 0]),c=0)
