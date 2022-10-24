import sys
# 依据：https://zhuanlan.zhihu.com/p/352724346

class Edge:
    __a:int = None # 顶点a
    b:int = None # 顶点b
    w:int = None # a->b距离

    def __init__(self, a, b, w):
        self.a = a
        self.b = b
        self.w = b


n, m, k = input("输入n m k:").split()
n, m, k = int(n), int(m), int(k)
edges: list[Edge]= []
for i in range(m):
    a, b, w = input("输入a b w:").split()
    a, b, w = int(a), int(b), int(k)
    edges.append(Edge(a, b, w))


def bellman_ford(edges:list[Edge], n:int, m:int , k:int)-> list[int]:
    """
    算法实现
    :param edges: 边集
    :param n: n个结点 结点索引为1~n
    :param m: m条边
    :param k: 表示从1到结点n至多k条边的最短距离
    :return: res 从结点1到n的不超过k条边的最短距离
    """
    res = [sys.maxsize for i in range(n + 1)]
    res[1] = 0
    for i in range(k):
        temp = res
        for j in range(m):
            a, b, w = edges[j].a, edges[j].b, edges[j].w
            res[b] = min(res[b], temp[a] + w)
    return res

res = bellman_ford(edges=edges, n=n, m=m, k=k)
print(res)
