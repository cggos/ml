import torch


a = torch.randn(2, 2)  # 缺失情况下默认 requires_grad = False
a = (a * 3) / (a - 1)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


x = torch.randn(3, 3, requires_grad=True)
print(x.grad_fn)

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x**2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

out.backward()
print(x.grad)

# 再来反向传播⼀一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
