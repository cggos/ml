#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
requirement: pip install ultralytics
"""

from ultralytics import SAM

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model = SAM("/home/gavin/Downloads/sam_b.pt")
model.info()  # display model information
model.predict("/home/gavin/Pictures/lawn_01.jpg")  # predict
