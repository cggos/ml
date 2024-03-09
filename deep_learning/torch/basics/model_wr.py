import torch
from torch import optim
from torch.optim import lr_scheduler
import torchvision.models as models


model = models.vgg16(pretrained=False)  # 加载模型
model_pretrained = models.vgg16(pretrained=True)  # 加载预训练模型和参数

model_path = "model.out.pth"

method = 2

if method == 0:
    torch.save(model, model_path)  # 保存和加载整个模型

    model = torch.load(model_path)
    # torch.load(model_path, map_location=torch.device("cpu"))  # 加载到 CPU
    # torch.load(model_path, map_location=lambda storage, loc: storage)  # 使用函数加载到 CPU
    # torch.load(model_path, map_location=lambda storage, loc: storage.cuda(1))  # 加载到 GPU 1
    # torch.load(model_path, map_location={"cuda:1": "cuda:0"})  # 从 GPU 1 加载到 GPU 0

    model.eval()

if method == 1:
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

if method == 2:
    n_epoch = 100
    optimizer = optim.SGD(model_pretrained.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_classification = torch.nn.CrossEntropyLoss()

    save_dict = {
        "epoch": n_epoch,
        "model": model_pretrained.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    torch.save(save_dict, model_path)

    model = torch.load(model_path, map_location="cpu")

    print(f"\nmodel state_dict items:\n")
    for key, value in model["model"].items():
        print(key, value.size(), sep="  ")
