#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : train_seg_net.py
@Site    : Vector-LabPics dataset: https://zenodo.org/record/3697452
@Author  : Gavin Gao
@Date    : 12/24/22 10:19 PM 
"""

import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.cuda.empty_cache()

Learning_Rate = 1e-5
width = height = 800  # image width and height
batchSize = 1

TrainFolder = "/dev_sdb/datasets/ml/LabPicsV1/Simple/Train/"
ListImages = os.listdir(os.path.join(TrainFolder, "Image"))

transformImg = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((height, width)),
        tf.ToTensor(),
        tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
transformAnn = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor()])


def ReadRandomImage():
    idx = np.random.randint(0, len(ListImages))  # Pick random image
    Img = cv2.imread(os.path.join(TrainFolder, "Image", ListImages[idx]))

    Filled = cv2.imread(
        os.path.join(
            TrainFolder, "Semantic/16_Filled", ListImages[idx].replace("jpg", "png")
        ),
        0,
    )
    Vessel = cv2.imread(
        os.path.join(
            TrainFolder, "Semantic/1_Vessel", ListImages[idx].replace("jpg", "png")
        ),
        0,
    )

    AnnMap = np.zeros(Img.shape[0:2], np.float32)  # Segmentation map
    if Vessel is not None:
        AnnMap[Vessel == 1] = 1
    if Filled is not None:
        AnnMap[Filled == 1] = 2
    Img = transformImg(Img)
    AnnMap = transformAnn(AnnMap)
    return Img, AnnMap


def LoadBatch():  # Load batch of images
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, height, width])

    for i in range(batchSize):
        images[i], ann[i] = ReadRandomImage()

    return images, ann


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(
    256, 3, kernel_size=(1, 1), stride=(1, 1)
)  # Change final layer to 3 classes
Net = Net.to(device)
optimizer = torch.optim.Adam(
    params=Net.parameters(), lr=Learning_Rate
)  # Create adam optimizer

for itr in range(10000):  # Training loop
    images, ann = LoadBatch()
    images = torch.autograd.Variable(images, requires_grad=False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    Pred = Net(images)["out"]  # make prediction
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get prediction classes
    print(itr, ") Loss=", Loss.data.cpu().numpy())
    if itr % 1000 == 0:
        print("Saving Model" + str(itr) + ".torch")
        torch.save(Net.state_dict(), str(itr) + ".torch")
