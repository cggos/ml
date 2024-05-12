#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import onnx

device = torch.device("cuda")
model = torch.load("/home/ghc/projects/ml/models/yolo/yolov5m-v7.0.pt", map_location=device)['model'].float()

input_names = ["input"]
output_names = ["output"]

torch.onnx.export(model, input_names, output_names, "model.onnx", opset_version=13, verbose=True)
