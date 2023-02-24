# 导入相关的torch工具包
import torch
import torchtext
# 导入torchtext.datasets中的文本分类任务
# from torchtext.datasets import text_classification
import os

load_data_path = './data/ag_news_csv'
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)

# 选取torchtext中的文本分类数据集ag_news即新闻主题分类，保存在指定目录下
# 并将数值映射后的训练和验证数据加载到内存中

train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root=load_data_path, split=('train', 'test'))

# train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)
# https://blog.csdn.net/qq_46092061/article/details/120598512

# 第一步: 构建带有Embedding层的文本分类模型.
# 第二步: 对数据进行batch处理.
# 第三步: 构建训练与验证函数.
# 第四步: 进行模型训练和验证.
# 第五步: 查看embedding层嵌入的词向量.


# 第一步: 构建带有Embedding层的文本分类模型
import torch.nn as nn
import torch.nn.functional as F

# 指定batch_size的大小
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextSentiment(nn.Module):
    """文本分类模型"""

    def __init__(self, vocab_size, embed_dim, num_class):
        """
        description: 类的初始化函数
        :param vocab_size: 整个语料包含的不同词汇总数
        :param embed_dim: 指定词嵌入的维度
        :param num_class: 文本分类的类别总数
        """
        super().__init__()
        # 实例化embedding层，传入三个参数，分别代表单词总数，词嵌入的维度，进行梯度求解时只更新部分权重
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        # 实例化全连接线性层，两个参数，输入输出的维度
        self.fc = nn.Linear(embed_dim, num_class)
        # 对定义的所有层执行初始化
        self.init_weights()

    def init_weights(self):
        """初始化权重函数"""
        # 指定初始权重的取值范围数
        initrange = 0.5
        # 各层权重使用均匀分布进行初始化
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化置为0
        self.fc.bias.data.zero_()

    def forward(self, text):
        """
        :param text: 文本数值映射后的结果
        :return: 与类别数尺寸相同的张量, 用以判断文本类别
        """
        # 获得embedding的结果embedded
        # >>> embedded.shape
        # (m, 32) 其中m是BATCH_SIZE大小的数据中词汇总数
        embedded = self.embedding(text)
        # 接下来我们需要将(m, 32)转化成(BATCH_SIZE, 32)
        # 以便通过fc层后能计算相应的损失
        # 首先, 我们已知m的值远大于BATCH_SIZE=16,
        # 用m整除BATCH_SIZE, 获得m中共包含c个BATCH_SIZE

        # 总行数对batch_size进行整除
        c = embedded.size(0) // batch_size
        # 之后再从embedded中取c*BATCH_SIZE个向量得到新的embedded
        # 这个新的embedded中的向量个数可以整除BATCH_SIZE
        embedded = embedded[:batch_size * c]
        # 因为我们想利用平均池化的方法求embedded中指定行数的列的平均数,
        # 但平均池化方法是作用在行上的, 并且需要3维输入
        # 因此我们对新的embedded进行转置并拓展维度
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        # 然后就是调用平均池化的方法, 并且核的大小为c
        # 即取每c的元素计算一次均值作为结果
        embedded = F.avg_pool1d(embedded, kernel_size=c)
        # 最后，还需要减去新增的维度, 然后转置回去输送给fc层
        return self.fc(embedded[0].transpose(1, 0))


# 实例化模型
# 获得整个语料包含的不同词汇总数
# vocab_size = len(train_dataset.get_vocab())
# print(vocab_size)

# 指定词嵌入维度
embed_dim = 32
num_class = len(train_dataset.getlabels())
