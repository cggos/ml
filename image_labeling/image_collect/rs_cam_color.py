#!/usr/bin/env python
# -*-coding:utf-8 -*-

import sys
import cv2

if __name__ == '__main__':
    img_dir = sys.argv[1]

    cap = cv2.VideoCapture(4)
    width = 640
    height = 480

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    i = 211
    while True:
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('s'):
            img_name = img_dir + "/rs_{:0>5d}.jpg".format(i)  # jpg for PASCAL VOC dataset format
            print("saved {}".format(img_name))
            cv2.imwrite(img_name, frame)
            i += 1
        cv2.imshow("capture", frame)
    cap.release()
    cv2.destroyAllWindows()
