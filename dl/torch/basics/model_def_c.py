import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=img_dim, out_features=128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 将输出值映射到0-1之间
        )

    def forward(self, x):
        return self.disc(x)


if __name__ == "__main__":
    batch_size = 32
    img_dim = 28 * 28 * 1
    input = torch.randn(batch_size, img_dim)

    disc = Discriminator(img_dim)
    output = disc(input)
    print(f"output.shape = {output.shape}")  # output.shape = torch.Size([32, 1])

    # 计算loss
    criterion = nn.BCELoss()
    loss1 = criterion(output, torch.ones_like(output))
    print(f"loss1 = {loss1}")  # loss1 = 0.714613676071167
    loss2 = criterion(output.view(-1), torch.ones_like(output.view(-1)))
    print(f"loss2 = {loss2}")  # loss2 = 0.714613676071167
