# while 基础应用
"""
while CONDITION:
    DO STH1
    DO STH2
    DO STH3
    while CONDITION2:
        DO STH1'
        DO STH2'
        DO STH3'
依然是按照缩进判断语句归属
"""
i = 0
while i < 5:
    while i != 4:
        i += 1
    print("caonima")
    i += 1
# print()不换行
i = 0
while i < 5:
    print(f"{i + 1} ", end='')
    i += 1
print()
# \t 可以多行数据对齐
print("caonima\tnimabi")
print("nigesha\tsuzhisanlian")

# for循环

"""
for VARIABLE in 要被处理的数据集（序列）:
    DO STH
无法构建无限循环
依然是根据缩进判断语句归属
"""

# for 处理字符串
name = "paidaxing"
for x in name:
    print(x, end='')
print()
# 序列类型 内容为可以一个个依次取出的类型
# range语句 range(num) 获取一个从0开始到num结束的序列，不含num
# range(num1, num2) 选取一个[num1, num2)区间的数字
# range(num1, num2, step) 选取一个[num1, num2)区间的数字，步长为step
for x in range(10):
    print(f"{x} ", end='')
print()
for x in range(5, 15):
    print(f"{x} ", end='')
print()
for x in range(0, 10, 2):
    print(f"{x} ", end='')
print()
# 规范上不建议，规则上可以在循环外部访问定义在循环内的k
for k in range(5):
    print(k)
# print(k)

# for 循环的嵌套和while一样
for k in range(3):
    print("关注塔菲喵")
    for j in range(2):
        print("关注塔菲谢谢喵", end='')
    print()
# break和continue
"""
continue直接进入下一次循环
break 直接跳出这个循环
"""