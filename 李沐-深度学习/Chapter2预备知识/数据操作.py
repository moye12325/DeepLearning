import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

# 改变x的形状。一维变二维
y = x.reshape(3, 4)
print(y)
print(y.shape)
print(y.numel())

zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3, 4)
print(zeros)
print(ones)

# 正态分布
z = torch.randn(3, 4)
print(z)
z1 = torch.randn(3, 4)
print(z1)
t1 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(t1)

# 张量运算
op1 = torch.tensor([1.0, 2, 4, 8])
op2 = torch.tensor([2, 2, 2, 2])
print(op1 + op2)
print(op1 - op2)
print(op1 * op2)
print(op1 / op2)
print(op1 ** op2)

# 张量连结
X = torch.arange(12).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
X1 = torch.cat((op1, op2), dim=0)
Y1 = torch.cat((op1, op2), dim=-1)
print(X1)
print(Y1)
a = torch.tensor([[[0, 1], [1, 1]], [[2, 2], [3, 2]]])
b = torch.tensor([[[0, 1], [1, 1]], [[2, 2], [3, 2]]])
print(torch.cat((a, b), dim=0))
print(torch.cat((a, b), dim=1))
print(torch.cat((a, b), dim=2))

print(a == b)
print(a.sum())

print("---------广播机制--------------")
# 广播机制  a复制列 b复制行 然后相加
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)
# 索引和切片
# 最后一个元素
print(x[-1])
# 第二到第三个元素 左开右闭
print(x[1:3])
# 写入元素到矩阵
X[2, 3] = 101
X[0:2, :] = 12

# 节省内存
a1 = torch.arange(12).reshape((3, 4))
b1 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(id(a1))  # 2122406153888
print(id(b1))  # 2122406153328
a = a + b
print(id(a))  # 2552922129648
# 由此可见是三个内存地址  +=运算不会新开辟内存 a1[:] = a1 + b1也不会新开辟内存
# a1 += b1
# print(id(a1)) #2548818894096
a1[:] = a1 + b1
print(id(a1))  # 2298562731120
c1 = torch.zeros_like(a1)
print(id(c1))  # 2181536472576
c1[:] = a1 + b1
print(id(c1))  # 2181536472576

# 转换为其他对象
print(type(X))  # <class 'torch.Tensor'>
A = X.numpy()
B = torch.tensor(A)
print(type(A))  # <class 'numpy.ndarray'>
print(type(B))  # <class 'torch.Tensor'>
