# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

# 看看label文件长啥样   确保能读取到数据
labels_dataframe = pd.read_csv('../data/Classification_Leaves/train.csv')
print(labels_dataframe.head(5))
print(labels_dataframe.describe())

leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
# 一共多少种树叶
print(n_classes)
# 前10个树叶标签
print(leaves_labels[:10])

# label放入字典 方便后面根据数字提取label
class_to_num = dict(zip(leaves_labels, range(n_classes)))
print(class_to_num)

# 再转换回来，方便最后预测的时候使用
num_to_class = {v: k for k, v in class_to_num.items()}
print(num_to_class)


class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_heights=256, resize_width=256):
        """

        :param csv_path: csv路径
        :param file_path:
        :param mode: 训练模式还是测试模式
        :param valid_ratio: 验证集比例
        :param resize_heights:
        :param resize_width:
        """
        self.resize_heights = resize_heights
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode

        self.data_info = pd.read_csv(csv_path, header=None)
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            self.train_labels = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            self.image_arr = self.train_image
            self.label_arr = self.train_labels
        elif mode == 'valid':
            self.vaild_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.vaild_image
            self.label_arr = self.valid_label

        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


train_path = '../data/Classification_Leaves/train.csv'
test_path = '../data/Classification_Leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '../data/Classification_Leaves/'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')
print(train_dataset)
print(val_dataset)
print(test_dataset)

# 加载数据
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


def im_convert(tensor):
    """展示数据"""
    image = tensor.to("cpu").clone.detch()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(val_loader)
inputs, classes = dataiter.next()

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    ax.set_title(num_to_class[int(classes[idx])])
    plt.imshow(im_convert(inputs[idx]))
plt.show()


# 看一下是在cpu还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()
print(device)


# 是否冻住模型前面的一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = True


# ResNet34
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft
