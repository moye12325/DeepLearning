import torch
import torch.nn as nn

# RNN参数：
# input_size：输入张量x的特征维度大小
# hidden_size：隐藏层张量h的特征维度大小
# num_layers：隐藏层层数
# nonlinearity: 激活函数的选择, 默认是tanh.

# 参数一: 输入张量的词嵌入维度5, 参数二: 隐藏层的维度(也就是神经元的个数), 参数三: 网络层数
rnn = nn.RNN(5, 6, 2, nonlinearity='relu')

# 参数一: sequence_length序列长度, 参数二: batch_size样本个数, 参数三: 词嵌入的维度, 和RNN第一个参数匹配
input = torch.randn(1, 3, 5)

# 参数一: 网络层数, 和RNN第三个参数匹配, 参数二: batch_size样本个数, 参数三: 隐藏层的维度, 和RNN第二个参数匹配
h0 = torch.randn(2, 3, 6)

output, hn = rnn(input, h0)
print(output)
print(output.shape)
print(hn)
print(hn.shape)
