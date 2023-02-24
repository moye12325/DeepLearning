from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
# t1 = classifier("I have been waiting for you")
#
# t2 = classifier([
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!"
# ])
#
# print(t1)
# print(t2)

# 目前可用的一些pipeline 有：
# feature-extraction 特征提取：把一段文字用一个向量来表示
# fill-mask 填词：把一段文字的某些部分mask住，然后让模型填空
# ner 命名实体识别：识别文字中出现的人名地名的命名实体
# question-answering 问答：给定一段文本以及针对它的一个问题，从文本中抽取答案
# sentiment-analysis 情感分析：一段文本是正面还是负面的情感倾向
# summarization 摘要：根据一段长文本中生成简短的摘要
# text-generation文本生成：给定一段文本，让模型补充后面的内容
# translation 翻译：把一种语言的文字翻译成另一种语言
# zero-shot-classification


# 1）Tokenizer
# 与其他神经网络一样，Transformer 模型不能直接处理原始文本，故使用分词器进行预处理。
# 使用AutoTokenizer类及其from_pretrained()方法。

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 若要指定我们想要返回的张量类型（PyTorch、TensorFlow 或普通 NumPy），
# 我们使用return_tensors参数

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs)
print(type(outputs))

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)