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
