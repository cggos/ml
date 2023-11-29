#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : pyg03.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 8:53 PM 
"""

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import TUDataset, ModelNet, ShapeNet
from torch_geometric import utils
from torch_geometric.data import Batch
from message_passing import ConnectedComponents
from dataset import COVID

ds = TUDataset(root='./data/', name='PROTEINS')
print(ds)
G = utils.to_networkx(ds[0])
nx.draw_kamada_kawai(G)
# plt.show()
data = Batch.from_data_list(ds[:10])
cc = ConnectedComponents()
count, perm = cc(data)
print(count)

ds = ModelNet(root="./data/ModelNet/")
print(ds)
m = utils.to_trimesh(ds[0])
m.show()

# TODO: COVID class error ?
covid = COVID(root='./data/COVID/')
print(covid)
G = utils.to_networkx(covid[0])
nx.draw_kamada_kawai(G)
# plt.show()
