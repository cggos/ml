#!/usr/bin/env python
# -*-coding:utf-8 -*-

import sys
import cv2

if __name__ == '__main__':
    video_dir = sys.argv[1]
    video_name = sys.argv[2]

    cap = cv2.VideoCapture(4)

    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    # width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_path = "{}/{}.mp4".format(video_dir, video_name)
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    flag = False
    while True:
        ret, frame = cap.read()
        if ret == True:
            k = cv2.waitKey(1)
            cv2.imshow('frame', frame)
            if flag:
                out.write(frame)
            if k == ord('s'):
                flag = True
                print("start recording video {}".format(video_path))
            elif k & 0xFF == ord('q') or k == 27:
                print("done")
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
