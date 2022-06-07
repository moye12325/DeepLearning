import torch

# x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([[1., 2.], [3., 4.]],requires_grad=True)
y = torch.ones(2, 2, requires_grad=True)
# print(y)
# z = x + y
# print(z)
# z = z.mean()
# print(z)
# z.backward()
# print(x.grad)

z1 = 2 * x + 3 * y
z1.backward(torch.ones_like(z1))
print(x.grad)
print(y.grad)
