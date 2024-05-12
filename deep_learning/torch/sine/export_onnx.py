import torch

import common.utils as utils
from net import Net

if __name__ == '__main__':
    model = Net(1, 10, 1)
    model.eval()

    utils.output_model_params(model, [1, 1, 1, 1])

    input_names = ["input"]
    output_names = ["output"]
    torch_input = torch.randn((1, 1, 1, 1))
    onnx_program = torch.onnx.export(
        model, torch_input, "model.onnx", input_names, output_names
    )
    # onnx_program.save("sine.onnx")
