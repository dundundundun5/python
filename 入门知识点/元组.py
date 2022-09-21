# 元组和列表一样，但一旦定义完成就不可修改
"""
变量名 = (元素，元素，元素)
变量名 = ()
变量名 = tuple()
"""

t1 = ()
t1 = tuple()
t1 = (1, "hello", False, 1, 1, False)
print(f"t1= {t1}, t1 type = {type(t1)}")
# 元组和列表一样可以嵌套
# 元组也可以通过下标获取元素
print(t1[-1])
# 元组只有 index count方法 len函数可以统计元组长度
print(t1.index("hello"))
print(t1.count(0))
print(len(t1))
# 元组的遍历和列表一致
for i in t1:
    print(f"{i} ", end='')
print()