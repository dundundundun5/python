# numpy快速入门
方法简介 为机器学习实战做铺垫
1. numpy.array(parameter...)

    array(LIST) 列表传参
    array(3, 2) 3行2列
2. numpy.zeros(parameter...)
    
    创建全0矩阵
3. numpy.ones(parameter...)

    创建全1矩阵
4. self.shape()
    
    获取矩阵大小
5. numpy.arange(low, high)
    
    创建递增的[low, high)数列
6. numpy.linspace(low, high, num)

    返回[low, high]之间等间距的num个数字
7. numpy.random.rand(parameter...)

    生成随机数矩阵
8. dtype=np.int32
    
    数据类型默认64为浮点数，但是可以通过dtype指定数据类型
9. numpy.dot(a, b)

    a 和 b 点乘
10. a @ b 

    a 和 b矩阵相乘
11. numpy数组已经重载了加减乘除