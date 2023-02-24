import torch
import torch.nn as nn

# input_size: 输入张量x中特征维度的大小.
# hidden_size: 隐层张量h中特征维度的大小.
# num_layers: 隐含层的数量.
# bidirectional: 是否选择使用双向LSTM, 如果为True, 则使用; 默认不使用.
rnn = nn.LSTM(5, 6, 2, bidirectional=False)

# 定义输入张量的参数含义: (sequence_length, batch_size, input_size)
input = torch.randn(1, 3, 5)

# 定义隐藏层初始张量和细胞初始状态张量的参数含义:
# (num_layers * num_directions, batch_size, hidden_size)
h0 = torch.randn(2 * 1, 3, 6)
c0 = torch.randn(2 * 1, 3, 6)

output, (hn, cn) = rnn(input, (h0, c0))
print(output)
print(output.shape)
print(hn)
print(hn.shape)
print(cn)
print(cn.shape)
