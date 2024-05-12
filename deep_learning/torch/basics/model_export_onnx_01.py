import torch.onnx
from models_diy import LeNet5

onnx_name = "model_out.onnx"
model = LeNet5()
# model = model.load_state_dict(torch.load("model.pth"))
# 导出模型前，必须调用model.eval()或者model.train(False)
model.eval()

batch_size = 1  # 随机的取值，当设置dynamic_axes后影响不大
dummy_input = torch.randn(batch_size, 1, 32, 32, requires_grad=True)
output = model(dummy_input)

torch.onnx.export(
    model,
    dummy_input,
    onnx_name,
    export_params=True,
    opset_version=15,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
