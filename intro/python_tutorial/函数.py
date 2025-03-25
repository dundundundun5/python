# 统计字符串长度
str1 = "caonima"
str2 = "nimabi"
str3 = "shabi"


def my_len(data):
    count = 0
    for i in data:
        count += 1
    print(f"{data}的长度为{count}")


my_len(str1)
"""
可以不用参数
def 函数名(参数1，参数2，参数3，...)
    函数体
    return 返回值
"""


# 定义一个函数


def say_hi():
    print("hi nihao")


# 调用函数
say_hi()

# None 是无返回值函数的实际返回内容
print(type(say_hi()))
# if中None等同于假

if say_hi():
    print("问候完毕")


# 函数说明文档


def add(x, y):
    """
    两数之和
    :param x: 相加数字1
    :param y: 相加数字2
    :return: 两数相加的结果
    """
    return x + y


# 函数嵌套调用 和循环一致，无需多解释

# 局部变量和全局变量的概念和其他编程语言一致


# def fun():
#     global a
#     a = 1
#     print(a)
#
#
# fun()
# print(a)
# 函数有多返回值


def my_return():
    return 1, [2, 3, 4]


a, b = my_return()


# 函数位置参数
# 和java传参一致

# 关键字参数
# 缺省参数和c++一样 默认值必须放最后

def multi(aa, bb, cc=114514):
    print(f"{aa}, {bb}, {cc}")


# 位置参数必须按照顺序
multi(cc=12, aa="caonima", bb="shabi")

multi(11, 12)


# 位置传递不定长参数
# args会作为元组存在接受不定长数量的参数


def fun2(*args):
    print(args)


fun2(1, 2, 3, "caonima", 1145.14)


# 关键字传递不定长参数 必须是键值对


def fun3(**kwargs):
    print(kwargs)


fun3(name="caonima", age=24, addr="下北泽")


# 函数作为参数传递
# 是计算逻辑的传递
# 和C++std::sort函数对象传递相似，同理Java的Arrays.sort()


def test_func(func3):
    result = func3(2, 4)
    print(result)


def compute(x, y):
    return x + y


test_func(compute)
# lambda匿名函数
# lambda 传入参数: 函数体(一行代码)
# 只能写一行代码

test_func(lambda x, y, z=2: x * y * z)
