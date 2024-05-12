import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, n_in, n_hidden, n_out) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        return x
