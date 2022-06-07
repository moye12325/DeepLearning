import torch

x = torch.ones(3, 4)
print(x)
y = torch.ones(3, 4)
print(y)

print(y.add(x))
print(y)

print(y.add_(x))
print(y)

