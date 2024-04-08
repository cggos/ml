import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

# from torchsummary import summary
from torchinfo import summary


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def show_imgs(img_raw, img_result):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_raw)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_result)
    plt.axis("off")
    plt.show()


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if "Normalize" in str(transform_train):
        norm_transform = list(
            filter(
                lambda x: isinstance(x, transforms.Normalize),
                transform_train.transforms,
            )
        )
        mean = torch.tensor(
            norm_transform[0].mean, dtype=img_.dtype, device=img_.device
        )
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype("uint8")).convert("RGB")
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype("uint8").squeeze())
    else:
        raise Exception(
            "Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(
                img_.shape[2]
            )
        )

    return img_


def output_model_params(net, input_shape):
    print("\n****************** net torchsummary ******************")
    summary(model=net, input_size=input_shape, device="cpu")

    print("\n****************** net.children ******************")
    for x in net.children():
        print(x)

    print("\n****************** net.named_children ******************")
    for name, x in net.named_children():
        print(f"{name}:\t {x}")

    print("\n****************** net.modules ******************")
    for x in net.modules():
        print(x)

    print("\n****************** net.named_modules ******************")
    for name, x in net.named_modules():
        print(f"{name}:\t {x}")

    print("\n****************** net.parameters ******************")
    params = list(net.parameters())
    print(params[-1].shape)

    print("\n****************** net.named_parameters ******************")
    for name, params in net.named_parameters():
        print(f"{name}:\t {params.shape}")

    print("\n****************** net.state_dict ******************")
    for k, v in net.state_dict().items():
        print(f"{k}:\t {v.shape}")
