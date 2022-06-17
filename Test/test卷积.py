import torch

X = torch.tensor([[19, 25],
                  [37, 43]])

Y = torch.tensor([[37, 47],
                  [67, 77]])

print(sum(X, Y))
