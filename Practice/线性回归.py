import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)
print(x)
y = 3 * x + 10 + 0.5 * torch.randn(x.size())


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = self.fc(x)
        return out


model = LinearRegression()

criterion = nn.MSELoss()  # 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=5e-3)  # 定义优化函数

# 迭代1000次是遍历整个数据集的次数。
# 先进性前向传播计算代价函数，后向传播计算梯度。
# 每次计算梯度前都要将梯度归零，否则会累计导致结果不收敛。
num_epochs = 1000  # 遍历整个训练集的次数
for epoch in range(num_epochs):
    # forward
    out = model(x)  # 前向传播
    loss = criterion(out, y)  # 计算损失函数
    # backward
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传递
    optimizer.step()  # 更新参数
    if (epoch + 1) % 20 == 0:  # 为了便于观察结果，每20次迭代输出当前的均方差损失。
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch + 1, num_epochs, loss.detach().numpy()))

# 模型测试
model.eval()  # 将模型由训练模式变为测试模式，将数据放入模型中进行预测。
y_hat = model(x)  # y_hat就是训练好的线性回归模型的预测值。
plt.scatter(x.numpy(), y.numpy(), label='原始数据')

# detach()用于停止对张量的梯度跟踪。模型训练阶段需要跟踪梯度，但模型的预测阶段不需要。
plt.plot(x.numpy(), y_hat.detach().numpy(), c='r', label='拟合直线')

plt.legend()
plt.show()

print(list(model.named_parameters()))
