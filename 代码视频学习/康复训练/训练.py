import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 下载数据集并对图片进行调整, 因为torchvision数据集的输出是PILImage格式, 数据域在[0, 1].
# 我们将其转换为标准数据域[-1, 1]的张量格式.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# 下载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 自定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 实例化模型
net = Net()
net.to(device)
# 损失与优化函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data中包含输入图像张量inputs，标签张量labels
        inputs, labels = data[0].to(device), data[1].to(device)

        # inputs, labels = data

        # 首先将优化器梯度归零
        optimizer.zero_grad()

        # 输入图像张量进入网络，得到输出张量outputs
        outputs = net(inputs)

        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, labels)

        # 反向传播、参数更新
        loss.backward()
        optimizer.step()

        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 200 == 0:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("Finished training")

# 设定模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # 打印原始图片
# imshow(torchvision.utils.make_grid(images))
# # 打印真实标签
# print('Groundtruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net2 = Net()
net2.load_state_dict(torch.load(PATH))

outputs = net2(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))

class_correct = list(0. for j in range(10))
class_total = list(0. for j in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net2(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for j in range(4):
            label = labels[j]
            class_total[label] += 1
            class_correct[label] += c[j].item()

for j in range(10):
    print('Acc of %5s : %2d %%' % (classes[j], 100 * class_correct[j] / class_total[j]))
