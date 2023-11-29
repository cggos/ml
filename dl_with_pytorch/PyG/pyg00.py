#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : pyg00.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 4:23 PM 
"""

import matplotlib.pyplot as plt
import sys

sys.stderr = sys.__stderr__
plt.rc('font', size=16)

import networkx as nx

G = nx.barabasi_albert_graph(100, 3)
_, axes = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'wspace': 0.5})
# nx.draw_kamada_kawai(G, ax=axes[0], node_size=120)
nx.draw(G, pos=nx.kamada_kawai_layout(G), ax=axes[0], node_size=120)
axes[1].imshow(nx.to_numpy_matrix(G), aspect='auto', cmap='Blues')
axes[0].set_title("$G$")
axes[1].set_title("$\mathbf{A}$")
plt.show()
