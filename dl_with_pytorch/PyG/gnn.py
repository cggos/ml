#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : gnn.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 9:46 PM 
"""

import torch
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_add_pool
from torch.nn import functional as F
from torch_geometric.data import Batch


class SimpleGNN(torch.nn.Module):
    def __init__(self, dataset, hidden=64, layers=6):
        super(SimpleGNN, self).__init__()
        self.dataset = dataset
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels=dataset.num_node_features, out_channels=hidden))
        for _ in range(1, layers):
            self.convs.append(GCNConv(in_channels=hidden, out_channels=hidden))
        self.jk = JumpingKnowledge(mode="cat")
        self.jk_lin = torch.nn.Linear(in_features=hidden * layers, out_features=hidden)
        self.lin_1 = torch.nn.Linear(in_features=hidden, out_features=hidden)
        self.lin_2 = torch.nn.Linear(in_features=hidden, out_features=dataset.num_classes)

    def forward(self, index):
        data = Batch.from_data_list(self.dataset[index])
        x = data.x
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x=x, edge_index=data.edge_index))
            xs.append(x)
        x = self.jk(xs)
        x = F.relu(self.jk_lin(x))
        x = global_add_pool(x, batch=data.batch)
        x = F.relu(self.lin_1(x))
        x = F.softmax(self.lin_2(x), dim=-1)
        return x
