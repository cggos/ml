import torch
import torch.nn as nn
import torchvision.models as models


model = models.vgg16(pretrained=False)  # 加载模型
model_pretrained = models.vgg16(pretrained=True)  # 加载预训练模型和参数

model_path = "model.out.pth"

if False:
    # 保存和加载整个模型
    torch.save(model, model_path)

    model = torch.load(model_path)
    model.eval()
else:
    # 仅仅保存模型参数以及分别加载模型结构和参数
    torch.save(model_pretrained.state_dict(), model_path)  # 模型参数保存

    print(f"keys:\n {model.state_dict().keys()}\n")

    weights = torch.load(model_path, map_location="cpu")  # 模型参数加载
    model = model.load_state_dict(weights)  # 加载模型参数到模型结构
    # model.eval()

    print(f"keys:\n {weights.keys()}\n")

    print(f"weights items:")
    for key, value in weights.items():
        print(f"{key}:\t {value.shape}")

    # print(f"\nmodel state_dict items:\n")
    # for key, value in model["state_dict"].items():
    #     print(key, value.size(), sep="  ")
