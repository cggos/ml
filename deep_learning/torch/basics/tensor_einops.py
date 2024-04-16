import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Rearrange
from einops import rearrange  # pip install einops
from einops import repeat
from einops import reduce

print(f"rearrange:")

i_tensor = torch.randn(16, 3, 224, 224)  # 在CV中很常见的四维tensor： （N，C，H，W）
print(i_tensor.shape)

o_tensor = rearrange(i_tensor, 'n c h w -> n h w c')
print(o_tensor.shape)

o_tensor = rearrange(i_tensor, 'n c h w -> n c (h w)')
print(o_tensor.shape)

o_tensor = rearrange(i_tensor, 'n c (m1 p1) (m2 p2) -> n c m1 p1 m2 p2', p1=16, p2=16)
print(o_tensor.shape)

print(f"\nrepeat:")

i_tensor = torch.randn(3, 224, 224)
print(i_tensor.shape)

o_tensor = repeat(i_tensor, 'c h w -> n c h w', n=16)
print(o_tensor.shape)

print(f"\nreduce:")

i_tensor = torch.randn((16, 3, 224, 224))
o_tensor = reduce(i_tensor, 'n c h w -> c h w', 'mean')
print(o_tensor.shape)
o_tensor_ = reduce(i_tensor, 'b c (m1 p1) (m2 p2)  -> b c m1 m2 ', 'mean', p1=16, p2=16)
print(o_tensor_.shape)


print(f"\nSequential:")

model = Sequential(
    Conv2d(3, 64, kernel_size=3),
    MaxPool2d(kernel_size=2),
    Rearrange('b c h w -> b (c h w)'),  # 相当于 flatten 展平的作用
    Linear(64 * 15 * 15, 120),
    ReLU(),
    Linear(120, 10),
)

i_tensor = torch.randn(16, 3, 32, 32)
o_tensor = model(i_tensor)
print(o_tensor.shape)

