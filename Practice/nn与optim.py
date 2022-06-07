# nn神经网络模块化接口
import torch
import torch.nn as nn


# 继承自父类nn.Module，包含网络层的定义以及forward方法
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        self.fc = nn.Linear(1, 1)  # 表示对模型的搭建，仅为一个全连接层fc，也叫线性层，（）的数字分别表示对输出和输出的维度
        # 其他层

    def forward(self, x):
        out = self.fc(x)
        return out


# 新建对象
net = net_name()

import torch.optim as optim

criterion = nn.MSELoss(reduction='none')


optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)

