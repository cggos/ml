# pip install tensorboard
# tensorboard --logdir=xxx/logs/

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter


class Cow(nn.Module):
    def __init__(self):
        super(Cow, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, data_input):
        data_output = self.sigmoid1(data_input)
        return data_output


data_input = torch.tensor([[1, -0.5], [-1, 3]])
data_input = torch.reshape(data_input, (-1, 1, 2, 2))
print(data_input.shape)

dataset = torchvision.datasets.CIFAR10(
    "./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True
)
dataloader = DataLoader(dataset, batch_size=64)


cow = Cow()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = cow(imgs)
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
