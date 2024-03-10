#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : seg_img.py
@Site    : ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
@Author  : Gavin Gao
@Date    : 12/24/22 4:30 PM 
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import models
import torchvision.transforms as T


def decode_segmap(image, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(model, image):
    # Apply the transformations needed
    trf = T.Compose(
        [
            T.Resize(256),  # 将图像尺寸调整为256×256
            # T.CenterCrop(224),  # 从图像的中心抠图，大小为224x224
            T.ToTensor(),  # 将图像转换为张量，并将值缩放到[0，1]范围
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  # 用给定的均值和标准差对图像进行正则化
    inp = trf(image).unsqueeze(0)

    # Pass the input through the net
    out = model(inp)["out"]
    print(out.shape)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(om.shape)
    print(np.unique(om))

    return decode_segmap(om)


# wget -nv https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/04/10/19/pinyon-jay-bird.jpg -O bird.png
img = Image.open("../data/bird.png")

# FCN
model = models.segmentation.fcn_resnet101(pretrained=True).eval()

# DeepLabV3 + MobileNet
# model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()

# try load local model
# model = torch.hub.load('../../', 'fcn_resnet101', '/home/ghc/projects/ml/models/fcn_resnet101_coco-7ecb50ca.pth', source='local')
# model = model.load_state_dict(torch.load('/home/ghc/projects/ml/models/fcn_resnet101_coco-7ecb50ca.pth'))

seg_rgb = segment(model, img)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(seg_rgb)
plt.axis("off")
plt.show()
