import torch
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from d2l import torch as d2l
from tqdm import tqdm
import numpy as np
from torch.utils import data
import wandb

NUM_SAVE = 50
net_list = "in->256->64"


class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 256)
        self.layer2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, X):
        X = F.relu(self.layer1(X))
        X = F.relu(self.layer2(X))
        return self.out(X)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 读取训练集 测试集
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
print("train_data and test_data shape", train_data.shape, test_data.shape)

# 去掉冗余数据
redundant_cols = ['Address', 'Summary', 'City', 'State']
for c in redundant_cols:
    del test_data[c], train_data[c]

# 数据预处理
large_vel_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount', 'Listed Price',
                  'Last Sold Price']
for c in large_vel_cols:
    train_data[c] = np.log(train_data[c] + 1)
    if c != 'Sold Price':
        test_data[c] = np.log(test_data[c] + 1)

# 把train和test去除id后放一起，train也要去掉label
all_features = pd.concat((train_data.iloc[:, 2:], test_data.iloc[:, 1:]))

# 时间数据赋日期格式
all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], format="%Y-%m-%d")
all_features['Last Sold On'] = pd.to_datetime(all_features['Last Sold On'], format="%Y-%m-%d")

for in_object in all_features.dtypes[all_features.dtypes == 'object'].index:
    print(in_object.ljust(20), len(all_features[in_object].unique()))

# 查询数字列 ->缺失数据赋0 -> 归一化
numeric_features = all_features.dtypes[all_features.dtypes == 'float64'].index
all_features = all_features.fillna(method='bfill', axis=0).fillna(0)
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

features = list(numeric_features)
features.extend(['Type', 'Bedrooms'])  # 加上类别数相对较少的Type, ,'Cooling features'
all_features = all_features[features]

print('before one hot code', all_features.shape)
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
print('after one hot code', all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
print('train feature shape:', train_features.shape)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
print('test feature shape:', test_features.shape)
train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float)
print('train label shape:', train_labels.shape)

criterion = nn.MSELoss()
in_features = train_features.shape[1]
net = MLP(in_features).to(device)


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(criterion(torch.log(clipped_preds),
                                torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    wandb.watch(net)
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in tqdm(range(num_epochs)):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        record_loss = log_rmse(net.to('cpu'), train_features, train_labels)
        wandb.log({'loss': record_loss, 'epoch': epoch})
        train_ls.append(record_loss)
        if (epoch % NUM_SAVE == 0 and epoch != 0) or (epoch == num_epochs - 1):
            torch.save(net.state_dict(), 'checkpoint_' + str(epoch))
            print('save checkpoints on:', epoch, 'rmse loss value is:', record_loss)
        del X, y
        net.to(device)
    wandb.finish()
    return train_ls, test_ls


k, num_epochs, lr, weight_decay, batch_size = 5, 2000, 0.005, 0.05, 256
wandb.init(project="kaggle_predict",
           config={"learning_rate": lr,
                   "weight_decay": weight_decay,
                   "batch_size": batch_size,
                   "total_run": num_epochs,
                   "network": net_list}
           )
print("network:", net)

train_ls, valid_ls = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)

# 使用现有训练好的net
net.to('cpu')
# 将网络应用于测试集。
preds = net(test_features).detach().numpy()

# 将其重新格式化以导出到Kaggle
test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
submission.to_csv('submission.csv', index=False)

net.to('cpu')
preds = net(test_features).detach().numpy()
# 将其重新格式化以导出到Kaggle
test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
submission.to_csv('submission.csv', index=False)

# 读取已有 继续进行训练
k, num_epochs, lr, weight_decay, batch_size = 5, 500, 0.0005, 0.08, 256
wandb.init(project="kaggle_predict",
           config={"learning_rate": lr,
                   "weight_decay": weight_decay,
                   "batch_size": batch_size,
                   "total_run": num_epochs,
                   "network": net_list}
           )
net.load_state_dict(torch.load('checkpoint_19676'))
print("network:", net)
net.to(device)
train_ls, valid_ls = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
net.to('cpu')
preds = net(test_features).detach().numpy()
# 将其重新格式化以导出到Kaggle
test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
submission.to_csv('submission.csv', index=False)

# 读取网络参数应用于测试集
net = []
net = MLP(test_features.shape[1])
net.load_state_dict(torch.load('checkpoint_250'))
net.to('cpu')
preds = net(test_features).detach().numpy()
# 将其重新格式化以导出到Kaggle
test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
submission.to_csv('submission.csv', index=False)

print(len(all_features['Type'].unique()))
print(len(all_features['Heating'].unique()))
print(len(all_features['Cooling'].unique()))
print(len(all_features['Parking'].unique()))
print(len(all_features['Bedrooms'].unique()))
print(len(all_features['Region'].unique()))
print(len(all_features['Elementary School'].unique()))
print(len(all_features['Middle School'].unique()))
print(len(all_features['High School'].unique()))
print(len(all_features['Flooring'].unique()))
print(len(all_features['Heating features'].unique()))
print(len(all_features['Cooling features'].unique()))
print(len(all_features['Appliances included'].unique()))
print(len(all_features['Laundry features'].unique()))
print(len(all_features['Parking features'].unique()))
print(len(all_features['City'].unique()))
