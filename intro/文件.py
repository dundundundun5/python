# 文件的读取操作
# open(name, mode, encoding)
# 文件名或者包含文件名的路径， 访问模式 r只读 w写入 a追加， 编码格式
# 最好使用关键字传参，因为参数较多
f = open("F:\\python\\intro\\文件.py", mode="r", encoding="UTF-8")
# 读取5个字符 读写指针后移5个字符
# print(f.read(5))
# 读取全部内容
# print(f.read())
# readlines() 读取所有行
# 读取文件 readline() 读取从当前读写指针开始的一行所有内容
# print(f.readline())
# for 循环读取文件
# for line in f:
#     print(line, end='')
# 文件需要停止读写 解除文件的占用
f.close()
# with open语法 会自动调用close方法
# with open("F:\\python\\intro\\文件.py", mode="r", encoding="UTF-8") as f:
#     for line in f:
#         print(line, end='')
# 清空并重新写入文件
with open(file="F:\\python\\intro\\test.txt", mode="w") as f:
    f.write("hello world")
    f.flush()# 刷新到硬盘里
# 不清空并追加写入文件
with open(file="F:\\python\\intro\\test.txt", mode="a") as f:
    f.write("hello world")
