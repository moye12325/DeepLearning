import torch

a = torch.cuda.is_available()
print(a)

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.rand(3, 3).cuda())

print("cuda是否可用: " + str(torch.cuda.is_available()))  # cuda是否可用
print("GPU的数量：" + str(torch.cuda.device_count()))  # GPU的数量
print("GPU的名称：" + str(torch.cuda.get_device_name(0)))  # GPU的名称
