import torch

from models_diy.lenet import LeNet5_00
from common.utils import LayerActivations

features = []
# def hook(module, input, output):
#     features.append(output.clone().detach())

net = LeNet5_00()
x = torch.randn(2, 1, 32, 32)

la = LayerActivations(net.C3)

y = net(x)
print(la.features)

la.remove()
