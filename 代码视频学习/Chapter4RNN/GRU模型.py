import torch
import torch.nn as nn

# 输入的维度、隐藏层的维度、隐藏层的层数
gru = nn.GRU(5, 6, 2)

# 序列的长度、batch_size批次样本的个数、输出张量的维度
input = torch.randn(1, 3, 5)

# 层数*方向数、batch_size、隐藏层的维度
h0 = torch.randn(2, 3, 6)

output, hn = gru(input, h0)
print(output)
print(output.shape)
print(hn)
print(hn.shape)
