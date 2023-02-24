import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数   每个图像由28*28个灰度像素值组成,图像共有10个类别。将每个图像视为具有784个输入特征 和10个类的简单分类数据集
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


# 激活函数 实现ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 模型   忽略了空间结构， 所以我们使用reshape将每个二维图像转换为一个长度为num_inputs的向量
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()

# 模型评估
d2l.predict_ch3(net, test_iter)