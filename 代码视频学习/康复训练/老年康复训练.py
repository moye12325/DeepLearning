import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络，输入通道维度1，输出通道维度6，卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第二层卷积神经网络，输入通道维度6，输出通道维度16，卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三层全连接网络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在(2,2)的池化窗口下执行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 计算size，除了第零个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

parameters = list(net.parameters())
print((len(parameters)))
# print(parameters)
print(parameters[0].size())