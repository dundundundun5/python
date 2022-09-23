# 类型注解告诉你传入什么类型，几个参数
# 变量的类型注解 变量名: 类型 = 值
# 或者用注释型# type: var_type

my_list: list[int] = [1, 2, 3]
my_tuple: tuple[str, int, int, int] = ("c", 1, 2, 3)
my_dict: dict[str, int]
v: bool = False


class Student:
    name: str = None
    age: int = 14
    gender = False # type: bool

    def fun(self, data : list):
        print(data)


# 参数和返回值注解的方法
# 注解只是提示，不是规定，依然可以在床上拉屎，在马桶上睡觉
def add(x: int, y: int) -> int:
    return x + y


stu: Student = Student()

# 联合类型的注解
from typing import Union

my_list: list[Union[str, int, float]] = [1, 2, "c", 114.514, "t"]
