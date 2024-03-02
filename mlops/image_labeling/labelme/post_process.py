import os
import numpy as np
import shutil
from PIL import Image
import cv2
import random


def makedir(path):
    os.makedirs(path) if not os.path.exists(path) else None


def copy_Data_label(P_dataset, out_path, Pimg, Plabel, Pviz):
    Pimg = os.path.join(out_path, Pimg)
    Plabel = os.path.join(out_path, Plabel)
    Pviz = os.path.join(out_path, Pviz)
    makedir(Pimg)
    makedir(Plabel)
    makedir(Pviz)

    for file in os.listdir(P_dataset):
        if not os.path.isdir(os.path.join(P_dataset, file)):
            continue
        try:
            shutil.copy(os.path.join(P_dataset, file, "img.png"), os.path.join(Pimg, file[:-5] + ".png"))
            shutil.copy(os.path.join(P_dataset, file, "label.png"), os.path.join(Plabel, file[:-5] + ".png"))
            shutil.copy(os.path.join(P_dataset, file, "label_viz.png"), os.path.join(Pviz, file[:-5] + ".png"))
        except:
            None


def convert_PIL_cv(out_path, Pinput, Poutput):
    Pinput = os.path.join(out_path, Pinput)
    Poutput = os.path.join(out_path, Poutput)
    makedir(Poutput)
    files = os.listdir(Pinput)
    # print(files)
    for file in files:
        print(os.path.join(Pinput, file))

        img = Image.open(os.path.join(Pinput, file))
        img_np = np.array(img)
        cv2.imwrite(os.path.join(Poutput, file), img_np)

        # ===========================================
        # print(np.unique(img_np))
        # img = cv2.imread(file, 0)
        # print(np.unique(img))
        # ===========================================


def gen_TrainValTest_file(root, image_path, label_path, txt_path):
    namelist = os.listdir(os.path.join(root, image_path))
    print(len(namelist))
    random.shuffle(namelist)

    makedir(os.path.join(root, txt_path))
    ftrain = open(os.path.join(root, txt_path, "train.lst"), "w")
    fval = open(os.path.join(root, txt_path, "val.lst"), "w")
    ftest = open(os.path.join(root, txt_path, "test.lst"), "w")

    for i in range(len(namelist)):
        img_file = os.path.join(image_path, namelist[i])
        label_file = os.path.join(label_path, namelist[i])
        string = "{} {}\n".format(img_file, label_file)
        """=============================================="""

        if i in range(6500): ftrain.write(string)
        if i in range(6500, 7000): fval.write(string)
        if i in range(7000, len(namelist)): ftest.write(string)


if __name__ == "__main__":
    P_dataset = "./annotations/"  # sh文件运行后的到的转换文件的总路径
    out_path = "./out/"  # 图片重整理的保存路径

    SVP_img = "data"  # 彩色图片
    SVP_label_pil = "label_pil"  # 标签图片的调色板模式
    SVP_viz = "viz"  # 彩色图片+标签可视化 路径
    SVP_label_cv = "label"  # opencv可正确读取的标签图片
    txt_path = "list"  # 神经网络使用train.txt/val.txt/test.txt

    # copy_Data_label(P_dataset, out_path, SVP_img, SVP_label_pil, SVP_viz)
    convert_PIL_cv(out_path, SVP_label_pil, SVP_label_cv)
    # gen_TrainValTest_file(out_path, SVP_img, SVP_label_cv, txt_path)
