# if基本格式
# if 条件:
#   条件成立时的结果
# 依靠缩进判断语句归属
age = 24
if age <= 114514:
    print("只有红茶可以吗？")
    print("只有意大利面可以吗？")

age = input("你多大了？\n")
# if else 组和用法
age = int(age)
if age <= 23:
    print("年龄太小")
    print("请离开这里")
else:
    print("只有红茶可以吗？")
# elif的使用

if age <= 10:
    print("caonima")
elif age <= 20:
    print("nimabi")
elif age <= 40:
    print("不是老毕登")
else:
    print("老了就要多登山强健身体")

# 判断语句的嵌套
name = "shabi"
height = input(f"{name}你好，请问你的身高是？")
height = int(height)
if name == "shabi":
    if height > 160:
        print("身材很结实啊？")
    else:
        print("您的身高符合正常范围")
    print("你就是歌姬吧")
    