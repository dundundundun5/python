# 字典创建一一对应关系 key-value
# 字典的定义
# 变量名 = {k : v, k : v, ...}
# my_dict = {}
# my_dict = dict()
my_dict = {"1": 12, "2": 14, "3": 15, 1: 3, 2: {77, 88, 99}}
print(my_dict)
# 从字典里获取数据
print(my_dict[2])
print(type(my_dict))
# 字典可以嵌套
my_dict = {
    "a": {
        "aa": 11,
        "bb": 22,
        "cc": 33
    },
    "b": {
        "aa": 44,
        "bb": 55,
        "cc": 66
    },
    "c": {
        "aa": 77,
        "bb": 88,
        "cc": 99
    }
}
print(my_dict["b"]["cc"])
# 新增
my_dict["d"] = {
        "aa": 111,
        "bb": 222,
        "cc": 333
}
print(my_dict)
# 更新
my_dict["a"] = {
    "aa": 1
}
print(my_dict)
# 删除
temp = my_dict.pop("d")
print(f"{temp} 被移除 {my_dict}")
# 清空元素
temp.clear()
# 获取全部key
print(my_dict.keys())
# 遍历字典的key
for key in my_dict.keys():
    print(key, my_dict[key])
# 字典的长度
print(len(my_dict))
