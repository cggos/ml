#!/usr/bin/env python
# -*-coding:utf-8 -*-

import sys
import cv2

videoCapture = cv2.VideoCapture('/tmp/111.mp4')

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

success, frame = videoCapture.read()
while success:
    cv2.imshow('windows', frame)
    cv2.waitKey(1000 // int(fps))
    success, frame = videoCapture.read()  # 获取下一帧

videoCapture.release()
