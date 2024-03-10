import torch

import models_diy
import common.utils as utils


net = models_diy.LeNet5()

print("\n****************** net ******************")
print(net)

print(f"\nnet dict keys\n: {net.state_dict().keys()}\n")

print(f"weights items:")
for key, value in net.state_dict().items():
    print(f"{key}:\t {value.shape}")

data_in = torch.rand(1, 1, 32, 32)
data_out = net(data_in)
print(f"input shape: {data_in.shape}")
print(f"output shape: {data_out.shape}")
print(f"output:\n {data_out}")

print("\n****************** net output_model_params ******************")
utils.output_model_params(net, data_in.shape)
