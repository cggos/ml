from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from net import Net


if __name__ == '__main__':
    x = np.arange(0.0, 1.0, 0.01)
    y = np.sin(2 * np.pi * x)
    x = x.reshape(100, 1)
    y = y.reshape(100, 1)
    plt.scatter(x, y)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    model = Net(1, 10, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 1000 == 0:
            print(f"After {epoch} iters, the loss is {loss.item()}")

    x = np.arange(0.0, 1.0, 0.005)
    x = x.reshape(200, 1)
    x = torch.Tensor(x)
    h = model(x)
    x = x.data.numpy()
    h = h.data.numpy()
    plt.scatter(x, h)
    plt.show()
