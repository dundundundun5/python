# 数据容器是一种可以容纳多份数据的数据类型，可以是任意类型

"""
定义格式
元素名 = [元素1， 元素2， ...]
元素名 = []
元素名 = list()
"""
name_list = ["caonima", "nimabi", True, 114514, 114.514]
print(name_list)
name_list = []
name_list = list()
# 嵌套列表
name_list = [[1, 2, 3], [4, 5, 6]]
print(name_list)
# 下标索引取出列表元素
print(name_list[0])
print(name_list[1])
# 负数索引 -1表示最后一个，-2类推
print(name_list[-2])
print(name_list[-1])
# 列表的元素数量
print(len(name_list))
# 取出嵌套列表中的某个
print(name_list[0][2])
# 列表可以插入、删除、清空、修改
name_list = ["caonima", "nimabi", True, 114514, 114.514]
# list 查询元素的索引
print(name_list.index(True))
# 修改索引的元素
name_list[1] = "shabbiness"
# 插入
name_list.insert(3, "nigeshabi")
print(name_list)
# 追加一个元素
name_list.append(24)
print(name_list)
# 追加一批元素
name_list2 = ["李田所", "下北泽"]
name_list.extend(name_list2)
print(name_list)
# 删除元素的两种方法
del name_list[1]
print(name_list)
element = name_list.pop(0)
print(f"{name_list}, {element}")
# 删除元素在列表中的第一个匹配项
name_list.remove(114514)
print(name_list)
name_list.append(True)
# 统计元素的数量
print(name_list.count(True))
# 清空列表
# name_list.clear()
# while遍历列表
idx = 0
while idx < len(name_list):
    print(f"{name_list[idx]}, ", end='')
    idx += 1
print()
# for 第一种遍历
for i in name_list:
    print(f"{i}, ", end='')
print()
# 列表统一赋值
initial_value, n = 11451, 4
l = []
l = [initial_value for i in range(n)]
print(l)