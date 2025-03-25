# 相当于java的动态绑定，即向上转型但动态绑定子类方法
class Animal:
    def say(self):
        pass


class Dog(Animal):
    def say(self):
        print("Dog")


class Cat(Animal):
    def say(self):
        print("Cat")

animal_list = [Dog(), Cat(), Animal()]

for i in animal_list:
    i.say()