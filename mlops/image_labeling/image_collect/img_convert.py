import os
import numpy as np
from PIL import Image
import cv2


def convert_img_to_jpg():
    path_in = "./"
    path_out = os.path.join("../", "imgs_jpg/")
    os.makedirs(path_out) if not os.path.exists(path_out) else None
    files = os.listdir(path_in)
    for file in files:
        ext = file.split('.')[-1]
        img_out = os.path.join(path_out, file[:-len(ext) - 1] + ".jpg")
        print(img_out)
        img_np = cv2.imread(os.path.join(path_in, file), -1)
        cv2.imwrite(img_out, img_np)


if __name__ == "__main__":
    convert_img_to_jpg()
