# 字符串的定义：单引 双引 三引号
name = 'caonima'
name_1 = "caonima"
name_2 = """caonima"""
# 包含单双引号的字符串
name_3 = "\""
name_4 = '"'
name_5 = """\'\""""
name_6 = "\\"
print(name_6)
# 字符串拼接 但不能拼接除了字符串的类型
ni = "ni"
print("cao " + ni + " ma")
# 字符串格式化 %用于表示占位符，多个占位符要括号括起来
# %s %d %f 字符串、整数、浮点数
msg = "我要说的是%s, 还要说的是%s" % ("caonima", "nimabi")
print(msg)
age = 24
namae = "李田所"
print("我的年龄是%d, 名字是%s" % (age, namae))
# 数字精度控制
# %m.nd %m.nf
# m控制宽度 n 控制小数点精度，会四舍五入()
print("%.2f" % 11.345)
# 快速格式化但是不限数据类不作精度控制
print(f"我的年龄是{age}, 名字是{namae}")
# 对表达式进行格式化+++++++++++++++++++++++++++++++++++++++
print("fuck you %s %s" % ("114" + "514", type(114514)))
# 字符串是字符的容器 同列表一样支持各种下表
my_str = "cao1ni1ma1le1ge1bi1ac"
value = my_str[10]
print(value)
# 字符串无法修改
# my_str[2] = "h"
# index()
print("ni的位置在", my_str.index("ni"))
# 字符串替换 其实是得到一个新的字符串
new_str = my_str.replace("ma", "**")
print(new_str)
# 字符串的分割，得到一个列表对象
new_list = my_str.split("1")
print(new_list)
# 字符串的掐头去尾
new_str = my_str.strip()
print(new_str)
# 取出前后的"ca" "ac"
new_str = my_str.strip("ca")
print(new_str)
# 计算某个字符的出现次数
print("a的出现次数是%d" % my_str.count("a"))
# len依然可以计算字符串长度
print(len(new_str))
