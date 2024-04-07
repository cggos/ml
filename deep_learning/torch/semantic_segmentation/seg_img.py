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

from common import dataset
from common import utils


def segment(model, image):
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

    return dataset.PASCAL_VOC.decode_segmap(om)


# wget -nv https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/04/10/19/pinyon-jay-bird.jpg -O bird.png
img = Image.open("./data/bird.png")

# FCN
# model = models.segmentation.fcn_resnet101(pretrained=True).eval()
model = models.segmentation.fcn_resnet101(
    weights=models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
).eval()

# DeepLabV3 + MobileNet
# model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()

# try load local model
# model = torch.hub.load('../../', 'fcn_resnet101', '/home/ghc/projects/ml/models/fcn_resnet101_coco-7ecb50ca.pth', source='local')
# model = model.load_state_dict(torch.load('/home/ghc/projects/ml/models/fcn_resnet101_coco-7ecb50ca.pth'))

seg_rgb = segment(model, img)

utils.show_imgs(img, seg_rgb)
