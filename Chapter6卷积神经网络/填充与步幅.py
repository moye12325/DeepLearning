import torch

from torch import nn


# 重定义一个计算卷积层的函数
# 初始化卷积层权重，对输入输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):

    X = X.reshape((1, 1) + X.shape) # 这里的（1，1）表示批量大小和通道数都是1
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=5, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

X = torch.rand(size=(8, 8))
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

print("--------------------步幅---------------------")
X = torch.rand(size=(8, 8))
conv2d = nn.Conv2d(1, 1, kernel_size=(3,5),padding=(0,1),stride=(3,4))
print(comp_conv2d(conv2d, X).shape)  # 取整