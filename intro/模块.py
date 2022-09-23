# [from 模块民] import 模块 [as 别名]
# import是必须存在的
from time import sleep

sleep(0)
import time

time.sleep(0)

from time import *
# 相当于import static

import math as nimabi

nimabi.sqrt(9)

# 自定义模块
# 创建py文件，文件名即模块名，文件里定义函数即可
"""
if __name__ == '__main__':
    代码块
用于在模块文件内部进行自我测试

__all__变量，用于使用*导入是构成访问限制
"""
if __name__ == '__main__':
    print("我在这里！")


