import torch

from models_diy.lenet import LeNet5_00
from common.utils import LayerActivations

net = LeNet5_00()
x = torch.randn(2, 1, 32, 32)

la = LayerActivations(net.C3)
_ = net(x)
la.remove()

print(la.features)
