#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : message_passing.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 9:35 PM 
"""

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor


class ConnectedComponents(MessagePassing):
    def __init__(self):
        super(ConnectedComponents, self).__init__(aggr="max")

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass

    def forward(self, data):
        x = torch.arange(data.num_nodes).view(-1, 1)
        last_x = torch.zeros_like(x)
        while not x.equal(last_x):
            last_x = x.clone()
        x = self.propagate(data.edge_index, x=x)
        x = torch.max(x, last_x)
        unique, perm = torch.unique(x, return_inverse=True)
        perm = perm.view(-1)
        if "batch" not in data:
            return unique.size(0), perm
        cc_batch = unique.scatter(dim=-1, index=perm, src=data.batch)
        return cc_batch.bincount(minlength=data.num_graphs), perm

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out
