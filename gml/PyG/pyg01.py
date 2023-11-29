#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : pyg01.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 4:44 PM 
"""

import torch

print(torch.__version__)

mat = torch.arange(12).view(3, 4)
print(mat)
print(mat[0])
print(mat[:, -1])
print(mat[:, 2:])
print(mat[:, ::2])
mat[:, ::2] = 42
print(mat)

rnd = torch.rand(3, 9)
print(rnd)
mask = rnd >= 0.5
print(mask)
print(mask.type())
print(rnd[mask])  # Masking returns always a 1-D tensor
print(rnd[:, (~mask).all(0)])

A = torch.randint(2, (5, 5))
print(A)
idx = A.nonzero().T
print(idx)
print(A[idx])  # ?
row, col = idx
print(A[row, col])

weight = torch.randint(10, (idx.size(1),))
print(weight)
A[row, col] = weight
print(A)
w, perm = torch.sort(weight)
print(w, idx[:, perm])
