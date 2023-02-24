import torch
from torch import nn
from d2l import torch as d2l

# 权重衰退是最广泛使用正则化的技术之一

#  生成一个人工的数据集。人工数据集的好处是很容易看到和真实值的区别
# 训练样本，测试样本，特征的维度
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 真实的权重和偏差
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 生成一个人工数据集
train_data = d2l.synthetic_data(true_w, true_b, n_train)
# 内存中如何读取一个数组 变成iter
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚

def l2_penalty(w):
    # 除以2是为了求导之后系数为1
    return torch.sum(w.pow(2)) / 2

# 定义训练函数
def train(lambd):
    # 初始化一个权重
    w, b = init_params()
    # 做了个很简单的线性回归和平方损失函数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 迭代的次数 和 学习率
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            # 唯一不一样 加上lambd
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
    d2l.plt.show()

# 没使用权重衰退
train(lambd=0)

# w的L2范数是： 12.827630996704102