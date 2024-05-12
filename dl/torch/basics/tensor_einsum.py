# https://www.cnblogs.com/qftie/p/16245124.html

import torch

i_a = torch.randn(16, 32, 4, 8)
i_b = torch.randn(16, 32, 8, 16)

out = torch.einsum('b h i j, b h j d -> b h i d', i_a, i_b)
print(out.shape)
