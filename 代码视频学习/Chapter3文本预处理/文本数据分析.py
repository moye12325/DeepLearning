import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置显示风格
plt.style.use('seaborn-bright')

# 读取两个文件
train_data = pd.read_csv('./data/train.tsv', sep="\t")
valid_data = pd.read_csv('./data/dev.tsv', sep="\t")

# 获得训练数据标签分布数量
sns.countplot("label", data=train_data)
plt.title("train_data")
plt.show()

# 获取验证数据标签分布数量
sns.countplot("label", data=valid_data)
plt.title("valid_data")
plt.show()

# 获取训练集 验证集的句子长度分布
# 在数据中添加句子长度列,每个元素的值都是对应句子列的长度
train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))

# 绘制句子长度列的数量分布图
sns.countplot("sentence_length", data=train_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布图
sns.distplot(train_data["sentence_length"])

# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()

valid_data["sentence_length"] = list(map(lambda x: len(x), valid_data["sentence"]))

# 绘制句子长度列的数量分布图
sns.countplot("sentence_length", data=valid_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布横坐标,不需要绘制纵坐标
sns.distplot((valid_data["sentence_length"]))
plt.yticks([])
plt.show()

# 绘制训练集 验证集长度分布的散点图
sns.stripplot(y='sentence_length', x='label', data=train_data)
plt.show()
sns.stripplot(y='sentence_length', x='label', data=valid_data)
plt.show()

# 获得训练集与验证集不同词汇总数统计
# 导入jieba用于分词
# 导入chain方法用于扁平化列表
import jieba
from itertools import chain

# 进行训练集 验证集的句子分词,并统计出不同词汇的总数
train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
valid_vocab = set(chain(*map(lambda x: jieba.lcut(x), valid_data["sentence"])))
print("训练集共包含不同词汇总数为: ", len(train_vocab), '\n', "验证集共包含不同词汇总数为: ", len(valid_vocab))

# 获得训练集上正负样本的高频形容词的词云
# 使用jieba中的词性标注
import jieba.posseg as pseg


def get_a_list(text):
    """用于获取形容词列表"""
    # 使用jieba的词性标注方法切分文本,获得具有词性属性flag和词汇属性word的对象,
    # 从而判断flag是否为形容词,来返回对应的词汇
    r = []
    for i in pseg.lcut(text):
        if i.flag == "a":
            r.append(i.word)
    return r


# 导入绘制词云的工具包
from wordcloud import WordCloud


def get_word_cloud(keyword_list):
    # 例化绘制词云的类, 其中参数font_path是字体路径, 为了能够显示中文,
    # max_words指词云图像最多显示多少个词, background_color为背景颜色
    wordcloud = WordCloud(font_path='./ttf/琉璃正楷.ttf', max_words=100, background_color='White')
    # 将传入的列表转换为词云生成器需要的字符串形式
    keyword_string = " ".join(keyword_list)
    # 生成词云
    wordcloud.generate(keyword_string)

    # 绘制图像并且显示
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# 获得训练集上的正样本
p_train_data = train_data[train_data["label"] == 1]["sentence"]
# 获得正样本每个句子上的形容词
train_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
get_word_cloud(train_p_a_vocab)

# 负样本
n_train_data = train_data[train_data["label"] == 0]["sentence"]
train_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_train_data))
get_word_cloud(train_n_a_vocab)
