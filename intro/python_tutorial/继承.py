# 继承语法 class Son(Dad):
class Phone:
    name = None
    producer = "HUAWEI"
    def say(self):
        print()

class Phone2:
    face_id = "114"


# 从左到右表示继承的类，较为靠左的类优先级较高
# 因此一旦有方法名一样，优先使用左边类方法
class Phone3(Phone, Phone2):
    # 覆盖父类的成员变量
    face_id = "514"

    def set(self):
        print(self.name, self.face_id, self.producer)

    # 覆盖父类的成员方法
    def say(self):
        print(self.face_id)

    # 调用父类成员变量和方法
    def src(self):
        print(super().face_id, self.face_id)