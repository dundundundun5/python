import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.use("tkagg")
def plot(arr, n, color):
    plt.xlim(0.5, n + 0.5)
    plt.ylim(0, n + 1)
    plt.yticks(range(0, n + 1, 1))
    plt.xticks(ticks=[i for i in range(1, n + 1)] ,labels=[str(i) for i in range(1, n+1)])
    plt.bar(x=arr, width=0.5,height=[i for i in range(1, n + 1)], color=color)
    plt.show()

def swap_permute(arr, n):
    """
    O(n!)
    permute_n表示n个数的全排列 arr_i表示数组元素
    当然也可以从第一个元素开始
    则permute_n =
    1. permute_n-1 APPEND arr_i
    2. (permute_n-2 APPEND arr_j) APPEND arr_i
    3. [(permute_n-3 APPEND arr_k) APPEND arr_j] APPEND arr_i
    :param arr: 最小字典序的排列数组
    :param n: 数组长度
    :return: 无
    """
    if n == 1:
        # 一个元素的全排列为其本身
        print(arr)
        return
    for i in range(n):
        arr[i], arr[n - 1] = arr[n - 1], arr[i]  # 将第i个元素放到最后
        swap_permute(arr, n - 1)  # 对其余元素（从i+1开始）做全排列
        arr[i], arr[n - 1] = arr[n - 1], arr[i]  # 将交换的数进行还原，保证序列不变
        # 为什么要还原，因为这是内排列，没有用额外的存储空间


def low_high_permute(arr, low, high):
    """
    O(n!)
    局部全排列
    :param arr: 数组
    :param low: 左端点索引
    :param high: 右端点索引
    :return: 无
    """
    if low == high:
        # 一个元素的全排列为其本身
        print(arr)
        return
    for i in range(low, high + 1, 1):
        arr[low], arr[i] = arr[i], arr[low]  # 把第i个元素放到左端
        low_high_permute(arr, low + 1, high)  # 对从 low + 1开始所有元素进行全排列
        arr[low], arr[high] = arr[high], arr[low]  # 还原，保持原排列不变


def neighbour_exchange_permute(arr, n):
    """
    就不做健壮性检查了

    1. 初始全排列1234...n，并规定其元素的方向都是向左的
    指向的元素比自己小，则处于活动状态
    2. 找到处于活动状态的最大元素x
    3. 推土机，反复与指向的元素交换，将大于x的元素方向倒转

    直至所有元素均为不活动元素
    第一个数向左，最后一个数向右，均不可移动

    :param arr:
    :param n:
    :return:
    """
    direction = [0 for i in range(n)]
    color = [0 for i in range(n)]
    def is_movable(index):
        if index == 0 and direction[index] == 0:
            return False
        if index == n - 1 and direction[index] == 1:
            return False
        if direction[index] == 0 and arr[index] < arr[index - 1]:
            return False
        if direction[index] == 1 and arr[index] < arr[index + 1]:
            return False
        return True

    def find_max_movable():
        max_active_value = -1
        max_active_index = -1
        for i in range(n):
            if is_movable(i) and arr[i] > max_active_value:
                max_active_value = arr[i]
                max_active_index = i
        return max_active_index

    # index-= 1 and index >= 1
    # index+= 1 and index <= n - 2
    def move(index):
        # 如果方向向左
        if direction[index] == 0:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            direction[index], direction[index - 1] = direction[index - 1], direction[index]
            return index - 1
        else:  # 如果方向向右
            arr[index], arr[index + 1] = arr[index + 1], arr[index]
            direction[index], direction[index + 1] = direction[index + 1], direction[index]
            return index + 1

    def reverse_direction(index):
        for i in range(n):
            if arr[i] > arr[index]:
                direction[i] = 0 if direction[i] == 1 else 1
    def all_unmovable():
        flag = True
        for i in range(n):
            # 只要有一个可移动
            if is_movable(i):
                flag = False
        return flag
    #print(arr)
    plot(arr,n, color)
    while True:
        if all_unmovable():
            break
        i = find_max_movable()
        i = move(i)
        reverse_direction(i)
        #print(arr)
        # 把比max大的所有元素移动方向反转
# https://www.cnblogs.com/1-2-3/archive/2011/05/23/generate-permutation-part3.html
if __name__ == '__main__':
    a = np.array([1, 2, 3])
    neighbour_exchange_permute(a, a.shape[0])
