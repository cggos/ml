#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : pyg02.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 7:36 PM 
"""

# pip install torch-geometric
# conda install pytorch-sparse -c pyg

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(data)

# pip install trimesh
# pip install "pyglet<2"
import trimesh

# wget -nc https://raw.githubusercontent.com/mikedh/trimesh/master/models/bunny.ply
m = trimesh.load('../bunny.ply')
# m.show()

from torch_geometric import utils

data = utils.from_trimesh(m)
print(data)

from torch_geometric import transforms

f2e = transforms.FaceToEdge(remove_faces=False)
print(f2e(data))

import networkx as nx
import matplotlib.pyplot as plt

G = nx.barabasi_albert_graph(n=100, m=3)
nx.draw_kamada_kawai(G)
data = utils.from_networkx(G)
print(data)
s2d = transforms.ToDense(num_nodes=120)
print(s2d(data))
adj = data.adj.numpy()
_, ax = plt.subplots(figsize=(10, 10))
ax.imshow(adj, cmap='Blues')
plt.show()
