# 集合 不支持元素的重复
# 变量名 = {元素，元素，元素，...}
my_set = {"cao", "ni", "ma", "ma", "le", "ge", "ge", "bi"}
my_set_empty = set()
my_set_empty = {}
print(my_set)
# 集合不可下标访问
# 集合存在修改方法
# 添加新元素
my_set.add("python")
my_set.add("cao")
print(my_set)
# 移除元素
my_set.remove("cao")
print(my_set)
# 随机取出一个元素
print(my_set.pop(), my_set)
# 清空元素
my_set_empty.clear()
# 差集 获取1有2没有的子集
my_set_2 = {"ni"}
print(my_set.difference(my_set_2))
# 消除差集 消除1里和2相同的元素
print(my_set.difference_update(my_set_2))
# 合并集合
print(my_set.union(my_set_2))
# 集合长度
print(len(my_set))
# 集合的遍历
for element in my_set:
    print(f"{element} ", end='')
print()