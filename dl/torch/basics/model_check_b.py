import torch
from torchvision.models import resnet50, ResNet50_Weights

model = torch.hub.load("pytorch/vision", "resnet50", weights="ResNet50_Weights.DEFAULT")

print(f"\nnet dict keys\n: {model.state_dict().keys()}\n")
