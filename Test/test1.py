import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

torch.cuda.is_available()

device = torch.device("cpu")
print(device)

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

for i in train_data.columns:
    if train_data[i].count() == 0:
        train_data.drop(labels=i, axis=1, inplace=True)

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

labels = train_data.iloc[:, 2]
print(labels)

train_data = train_data.drop('Sold Price', 1)

print(train_data.shape)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将以下参数不计入特征
all_features = all_features.drop('Address', 1)
all_features = all_features.drop('Summary', 1)
all_features = all_features.drop('Heating', 1)
all_features = all_features.drop('Cooling', 1)
all_features = all_features.drop('Parking', 1)
all_features = all_features.drop('Bedrooms', 1)
all_features = all_features.drop('Region', 1)
all_features = all_features.drop('Elementary School', 1)
all_features = all_features.drop('Middle School', 1)
all_features = all_features.drop('High School', 1)
all_features = all_features.drop('Flooring', 1)
all_features = all_features.drop('Heating features', 1)
all_features = all_features.drop('Cooling features', 1)
all_features = all_features.drop('Appliances included', 1)
all_features = all_features.drop('Laundry features', 1)
all_features = all_features.drop('Parking features', 1)
all_features = all_features.drop('Listed On', 1)
all_features = all_features.drop('Last Sold On', 1)
all_features = all_features.drop('State', 1)
all_features = all_features.drop('Lot', 1)

print("--------------------all_features---------------------------")
print(all_features.shape)

print(all_features.iloc[0:4, [1, -7, -2, -1]])

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)  # 对非数值进行one - hot处理
print("--------------------one - hot---------------------------")
print(all_features.shape)

# 从pandas提取Numpy格式，转换为张量
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(labels, dtype=torch.float32)
print("train_features.shape :", train_features.shape)
print("test_features.shape :", test_features.shape)

# 定义损失 带有损失平方的线性模型
loss = nn.MSELoss()  # 均方损失函数
in_features = train_features.shape[1]  # 特征数


# 三层神经网络模型
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128),
                        nn.ReLU(), nn.Linear(128, 1))
    return net


# 相对误差
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    clipped_preds = torch.squeeze(clipped_preds, -1) # 降维
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            # X = X.to(device)
            # y = y.to(device)
            l = loss(net(X).squeeze(-1), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 折交叉验证过程中返回第i折的数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# K折交叉验证中训练K次后，返回训练和验证误差的平均值。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # net.to(device)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.05, 0, 128

train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

test_features = test_features.to(device)


# 最后一次完整训练以提交
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    # 将网络所有成员、函数、操作都搬移到device上面
    net.to(device)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # preds = net(test_features).detach().numpy()
    preds = net(test_features)
    preds = preds.cpu()
    preds = preds.detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
