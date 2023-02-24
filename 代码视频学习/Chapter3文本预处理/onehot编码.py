import joblib

from keras.preprocessing.text import Tokenizer

# 进行one-hot编码

# 假定vocab为语料集所有不同词汇的集合
vocab = {'吴亦凡', '李易峰', '王力宏', '罗志祥', '李云迪'}
# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)
# 使用词汇映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0] * len(vocab)
    # 使用映射器转化现有文本数据，每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为：", zero_list)

# 使用joblib保存映射器，以便后续使用
tokenizer_path = './Tokenizer'
joblib.dump(t, tokenizer_path)

# onehot编码器的使用:

# 导入用于对象保存与加载的joblib
# from sklearn.externals import joblib
# 加载之前保存的Tokenizer, 实例化一个t对象
t = joblib.load(tokenizer_path)

# 编码token为李易峰
token = "李易峰"
# 使用t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个zero_list
zero_list = [0] * len(vocab)
# 令zero_list的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为：", zero_list)
