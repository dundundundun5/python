# 专注py异常捕获的语法
"""
try:
    代码块
except:
    执行代码
"""
# 捕获异常
try:
    print(1 / 0)
except:
    print("你就是个几把")
# 捕获具体的异常
try:
    print(1 / 0)
except ZeroDivisionError as e:
    print(e)
# 捕获多个异常
try:
    print(name, 1 / 0)
except (NameError, ZeroDivisionError) as e:
    print(e)
# 捕获所有的异常
# else 表示 没有发生异常的执行内容
# Exception表示所有异常之父
# finally表示无论如何都一定会执行的代码块
try:
    print("1")
except Exception as e:
    print(e)
else:
    print("没有异常")
finally:
    print("我百分百执行")
# 异常的传递性 虽然不需要写throws
# 总有一层函数要处理异常
# 和java异常栈的规律基本一致