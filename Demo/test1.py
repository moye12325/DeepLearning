import torch

a = torch.arange(12).reshape(3, 4)
b = torch.arange(3).reshape(1, 3)

print(a)
print(b)

print(a + b)
