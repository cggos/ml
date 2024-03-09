import torch
from torchvision.models import resnet50, ResNet50_Weights

torch.hub.list("pytorch/vision")

print(torch.hub.help("pytorch/vision", "deeplabv3_resnet101"))

model = torch.hub.load("..", "custom", "../weights/best.pt", source="local")

torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

torch.hub.load("pytorch/vision", "resnet50", weights="ResNet50_Weights.DEFAULT")

torch.hub.load("pytorch/vision", "deeplabv3_resnet101", pretrained=True)

torch.hub.load("pytorch/vision:v0.4.2", "deeplabv3_resnet101", pretrained=True)

# 如果模型的发布者后续加入错误修复和性能改进，用户也可以非常简单地获取更新，确保自己用到的是最新版本
torch.hub.load("pytorch/vision", "deeplabv3_resnet101", force_reload=True)

# 稳定性更加重要，他们有时候需要调用特定分支的代码
torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN", pretrained=True)

# 查看模型可用方法
dir(model)
help(model.forward)
