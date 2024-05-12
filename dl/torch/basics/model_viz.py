#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Ref
    * https://zhuanlan.zhihu.com/p/232348083
"""

import torch
from torchviz import make_dot

from dogs_vs_cats import network_model
from models.model_diy import ModelTest


if __name__ == "__main__":
    model0 = ModelTest()
    model1 = network_model.Net()

    data_in = torch.rand(1, 3, 4, 4)
    out0 = model0(data_in)

    print(f"model0 out:\n {out0}")

    method = 1
    if method == 0:
        torch.save(model0, "model.out.pt")  # netron
    elif method == 1:
        g = make_dot(out0)  # graphviz
        g.render("model.out", view=True)
        # g.view()
    # elif method == 2:
    #     # 1. 来用tensorflow进行可视化
    #     from tensorboardX import SummaryWriter
    #     with SummaryWriter("./log", comment="sample_model_visualization") as sw:
    #         sw.add_graph(model, sampledata)
