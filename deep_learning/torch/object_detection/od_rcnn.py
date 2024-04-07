#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : od_rcnn.py
@Site    : ref: https://learnopencv.com/faster-r-cnn-object-detection-with-pytorch
@Author  : Gavin Gao
@Date    : 12/24/22 5:46 PM 
"""

import os
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Agg')
import torchvision
import torchvision.transforms as T
from torchvision import models

# 把当前文件所在文件夹的父文件夹路径加入到 PYTHONPATH
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import dataset
from common import utils


def get_prediction(img, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    """
    # img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    imgT = transform(img).unsqueeze(0)
    pred = model(imgT)
    pred_class = [
        dataset.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]["labels"].numpy())
    ]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]["boxes"].detach().numpy())
    ]
    pred_score = list(pred[0]["scores"].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    return pred_boxes, pred_class


# Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
# to ~/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
model = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
).eval()

# wget https://www.wsha.org/wp-content/uploads/banner-diverse-group-of-people-2.jpg -O people.jpg
img_path = "./data/banner-diverse-group-of-people-2.jpg"
threshold = 0.8

img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
boxes, pred_cls = get_prediction(img_raw, threshold)

rect_th = 3
text_size = 3
text_th = 3
img = img_raw.copy()
for i in range(len(boxes)):
    pt0 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
    pt1 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
    cv2.rectangle(img, pt0, pt1, color=(0, 255, 0), thickness=rect_th)
    cv2.putText(
        img,
        pred_cls[i],
        pt0,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        (0, 255, 0),
        thickness=text_th,
    )

utils.show_imgs(img_raw, img)
