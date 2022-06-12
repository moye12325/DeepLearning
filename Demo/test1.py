import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
# 每次随机读256张图片。返回训练的iter和测试的iter
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 我们读出来的图片都是长和宽都是长28的图片，通道数为1，是一个3d的格式
# 对于softmax来说需要输入的是一个向量
# 所以需要将图片拉长，拉成一个向量。拉向量会损失掉很多空间信息
# 28*28=784
# 将展平每个图像，把它们看作长度为784的向量。 因为我们的数据集有10个类别，所以网络输出维度为 10

num_inputs = 784
num_outputs = 10

# 形状是它的行数784和列数10，requires_grad=True需要检测梯度
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 偏移是一个长为10的向量
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

# 这里就是对于softmax公式的实现
# X看成以784行、10列举证的每一行
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition #这里应用了广播机制

# 这里是验证softmax函数是否正确
# X为两行5列的一个X
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))

# 实现模型
# 我们需要的是一个批量大小乘以输入维数的一个矩阵。-1指的是批量大小(-1是指根据列数自动计算行数)，W.shape[0]是784.
# batch_size是256.所以X会变成一个256*784的矩阵。
# 然后将W和X进行矩阵乘法+b
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 接着实现交叉熵损失
# 创建一个数据y_hat，其中包含2个样本在3个类别的预测概率， 使用y作为y_hat中概率的索引
# 首先要实现 在预测值里面，怎么样，根据标号把对应的预测值拿出来
# 例子
# 创建一个长度为2的向量，它是一个整数型
y = torch.tensor([0, 2])
# y_hat就是预测值，假设我们有3类，对2个样本做预测，那就是一个2*3的矩阵
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])

# 实现交叉商损失函数
# 给定预测和真实的y
# 就是根据交叉熵损失函数的定义实现的函数
def cross_entropy(y_hat, y):
    # 首先生成一个range(len(y_hat)的向量，y是拿出来真实标号的预测值
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))


# 因为我们做的是分类问题，所以将预测类别与真实类别y元素进行比较看是不是正确的？
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # y_hat是一个二维矩阵的话，它的shape大于1且它的列也大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 这里的axis应该是同理于numpy axis=1即跨列 也即行的方向
        # 求出元素值最大的那个下标
        y_hat = y_hat.argmax(axis=1)
    # 然后作比较
    cmp = y_hat.type(y.dtype) == y
    # 转成和y一样的形状
    return float(cmp.type(y.dtype).sum())

# accuracy(y_hat, y)找出来正确预测的样本数除以一共的样本数即是正确的概率
print(accuracy(y_hat, y) / len(y))

# 评估在任意模型net的准确率
# 给一个模型，给定一个迭代器。计算数据迭代器上的精度
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#
# if __name__ == "__main__":
#     print(evaluate_accuracy(net, test_iter))


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 创建一个长度为3的迭代器类累加一些信息。训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            # 做一次更新
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    # metric[0]所有loss的累加除以样本数。metric[1]所有正确的样本数除以总样本数
    return metric[0] / metric[2], metric[1] / metric[2]

# 定义一个在动画中绘制数据的实用程序类
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        d2l.plt.draw()
        d2l.plt.pause(0.001)
        d2l.plt.draw()


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    # 扫n边数据
    for epoch in range(num_epochs):
        # 扫一次更新我们的模型得到训练的误差train_metrics
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 测试数据集上评估精度
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


if __name__ == "__main__":
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()