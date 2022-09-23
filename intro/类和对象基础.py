class Student:
    name = None
    gender = None
    age = None

    def say(self, msg):
        print("do nothing", msg)

    def ring(self):
        import winsound
        winsound.Beep(2000, 1000)

    def __init__(self, name="李田所", age=24, gender="男"):
        """
        构造方法不再是类名(java/c++) 而是__init__

        :param name:
        :param age:
        :param gender:
        """
        self.name = name
        self.age = age
        self.gender = gender

    def __str__(self):
        """
        str魔术方法 用于将对象转为字符串时自动调用
        类似java的toString()
        :return:
        """
        return f"{self.name}, 今年{self.age}，{self.gender}"

    def __lt__(self, other):
        """
        lt魔术方法 用于重载小于符号 大于符号一并重载
        :param other:
        :return:
        """
        return self.age < other.age

    def __le__(self, other):
        """
        用于重载小于等的符号 大于等于一并重载
        :param other:
        :return:
        """
        return self.age <= other.age

    def __eq__(self, other):
        """
        未重载==之前，==比较对象的内存地址
        重载后按要求比较
        相当于equal
        :param other:
        :return:
        """
        return self.age == other.age
# 创建一个对象
stu_1 = Student()
stu_1.name = "caonima"
stu_1.gender = "男"
stu_1.say("caonima")
# stu_1.ring()
# 类定义的格式和其他面向对象语言基本一致

# 但是必须传入self参数，表示调用方法的当前对象，相当于java的this

# 构造方法是__init__

# 魔术方法，是内置的
print(stu_1)
# 私有成员变量和方法，标识符需要用两个下划线开头
class Phone:
    __current_voltage = None

    def __say(self):
        print()
# 双下划线相当于private 编程思想与C++、Java保持一致