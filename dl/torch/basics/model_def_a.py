import torch
import torch.nn as nn
from collections import OrderedDict


relu = nn.ReLU(inplace=True)
input = torch.randn(2, 3)
print(input)
output = relu(input)
print(output)  # 小于0的都被截断为0
# 等价于input.clamp(min=0)

net1 = nn.Sequential()
net1.add_module("conv", nn.Conv2d(3, 3, 3))
net1.add_module("batchnorm", nn.BatchNorm2d(3))
net1.add_module("activation_layer", nn.ReLU())

net2 = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU())

net3 = nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(3, 3, 3)),
            ("bn1", nn.BatchNorm2d(3)),
            ("relu1", nn.ReLU()),
        ]
    )
)

print(f"net1:\n{net1}\n")
print(f"net2:\n{net2}\n")
print(f"net3:\n{net3}\n")

print(f"net1 conv:\t{net1.conv}")
print(f"net2 [0]:\t{net2[0]}")
print(f"net3 conv1:\t{net3.conv1}")

input = torch.randn(1, 3, 4, 4)
output1 = net1(input)
output2 = net2(input)
output3 = net3(input)
output4 = net3.relu1(net1.batchnorm(net1.conv(input)))
print(f"output1:\n{output1}\n")
print(f"output2:\n{output2}\n")
print(f"output3:\n{output3}\n")
print(f"output4:\n{output4}\n")

modellist = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
input = torch.randn(1, 3)
for model in modellist:
    input = model(input)
print(f"modellist output:\n{input}\n")
