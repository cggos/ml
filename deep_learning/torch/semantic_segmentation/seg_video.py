# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T


def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:
    """Visualize segmentation mask.

     Parameters
     ----------
     img: numpy.ndarray
         Image with shape `(H, W, 3)`.
     masks: numpy.ndarray
         Binary images with shape `(N, H, W)`.
     colors: numpy.ndarray
         color for mask, shape `(N, 3)`.
         if None, generate random color for mask
     alpha: float, optional, default 0.5
         Transparency of plotted mask

     Returns
     -------
     numpy.ndarray
         The image plotted with segmentation masks, shape `(H, W, 3)`

     """
    if colors is None:
        colors = np.random.random((masks.shape[0], 3)) * 255
    else:
        if colors.shape[0] < masks.shape[0]:
            raise RuntimeError(
                f"colors count: {colors.shape[0]} is less than masks count: {masks.shape[0]}"
            )
    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)

    return img.astype(np.uint8)


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

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
    trf = T.Compose([T.ToPILImage(),
                     # T.Resize(256),  # 将图像尺寸调整为256×256
                     # T.CenterCrop(224),  # 从图像的中心抠图，大小为224x224
                     T.ToTensor(),  # 将图像转换为张量，并将值缩放到[0，1]范围
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 用给定的均值和标准差对图像进行正则化
    inp = trf(image).unsqueeze(0)

    # Pass the input through the net
    out = model(inp)['out']
    print(out.shape)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(om.shape)
    print(np.unique(om))

    return decode_segmap(om)


print(cv2.__version__)
cap = cv2.VideoCapture(2)

if not (cap.isOpened()):
    print("Could not open video device")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()

while (True):
    ret, frame = cap.read()

    seg_rgb = segment(model, frame)

    print("shape:" + str(frame.shape))
    print(type(frame))

    print("shape:" + str(seg_rgb.shape))
    print(type(seg_rgb))

    result = cv2.addWeighted(frame, 0.6, np.asarray(seg_rgb), 0.5, 0)

    cv2.imshow('preview', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
