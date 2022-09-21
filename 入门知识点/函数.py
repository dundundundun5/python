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


def fun():
    global a
    a = 1
    print(a)


fun()
print(a)
