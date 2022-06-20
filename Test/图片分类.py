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
            self.label_arr = self.train_label
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
