import numpy as np
import torch

# 标量由只有⼀个元素的张量表⽰
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y,
      x * y,
      x / y,
      x ** y)

# 可以将向量视为标量值组成的列表,将这些标量值称为向量的元素（element）或分量（component）。x = torch.arange(4)
X1 = torch.arange(4)
print(X1)

# 向量只是⼀个数字数组，就像每个数组都有⼀个⻓度⼀样，每个向量也是如此。
print(len(X1))
print(x.shape)
print(X1.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)  # B为对称矩阵

# 向量是⼀阶张量，矩阵是⼆阶张量。张量⽤特殊字体的⼤写字⺟表⽰（例如，X、Y和Z），它们的索引机制与矩阵类似。
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 张量运算
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:', A)
print('B:', B)
print('A + B:', A + B)  # 矩阵相加
print('A * B:', A * B)  # 矩阵相乘

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('X:', X)
print('a + X:', a + X)  # 矩阵的值加上标量
print('a * X:', a * X)
print((a * X).shape)

print('------------矩阵的sum运算-------------')
print('A:', A)
print('A.shape:', A.shape)
print('A.sum():', A.sum())
print('A.sum(axis=0):', A.sum(axis=0))  # 沿0轴汇总以生成输出向量
print('A.sum(axis=1):', A.sum(axis=1))  # 沿1轴汇总以生成输出向量
print('A.sum(axis=1, keep dims=True)', A.sum(axis=1, keepdims=True))  # 计算总和保持轴数不变，保持维度不变
print('A.sum(axis=[0, 1]):', A.sum(axis=[0, 1]))  # Same as `A.sum()`
print('A.mean():', A.mean())
print('A.sum() / A.numel():', A.sum() / A.numel())

print('-----------向量-向量相乘（点积）-----------------')
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print('x:', x)
print('y:', y)
print('向量-向量点积:', torch.dot(x, y))

print('---------------矩阵-向量相乘(向量积)--------------')
print('A:', A)  # 5*4维
print('x:', x)  # 4*1维
print('torch.mv(A, x):', torch.mv(A, x))

print('---------------矩阵-矩阵相乘(向量积)---------------')
print('A:', A)  # 5*4维
B = torch.ones(4, 3)  # 4*3维
print('B:', B)
print('torch.mm(A, B):', torch.mm(A, B))

print('----------------范数--------------------------')
u = torch.tensor([3.0, -4.0])
print('向量的𝐿2范数:', torch.norm(u))  # 向量的𝐿2范数
print('向量的𝐿1范数:', torch.abs(u).sum())  # 向量的𝐿1范数
v = torch.ones((4, 9))
print('v:', v)
print('矩阵的𝐿2范数:', torch.norm(v))  # 矩阵的𝐿2范数

print('-------------------根据索引访问矩阵---------------')
y = torch.arange(10).reshape(5, 2)
print('y:', y)
index = torch.tensor([1, 4])
print('y[index]:', y[index])

print('-----------------理解pytorch中的gather()函数------------')
# https://blog.csdn.net/weixin_42899627/article/details/122816250函数解释。越学越迷糊
a = torch.arange(15).view(3, 5)
print('二维矩阵上gather()函数')
print('a:', a)
b = torch.zeros_like(a)
b[1][2] = 1  ##给指定索引的元素赋值
b[0][0] = 1  ##给指定索引的元素赋值
print('b:', b)
c = a.gather(0, b)  # dim=0
d = a.gather(1, b)  # dim=1
print('c:', c)
print('d:', d)

