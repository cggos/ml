# @ref: https://zhuanlan.zhihu.com/p/457851552

import torch
import torch.nn as nn
import numpy as np


class _Cbr(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        return input

    @staticmethod
    def symbolic(g, input):
        return g.op("onnx_test::cbr", *[input], **{})


def Cbr(input):
    return _Cbr.apply(input)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = Cbr(x)
        x = Cbr(x)
        return x


model = Model()
model.eval()

x = torch.randn((1, 3, 12, 12))

torch.onnx.export(
    model,
    x,
    'model_revised.onnx',
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={
        'input': {0: 'batch', 2: 'h', 3: 'w'},
        'output': {0: 'batch', 2: 'h', 3: 'w'},
    },
)
