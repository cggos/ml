import os
import random


def makedir(path):
    os.makedirs(path) if not os.path.exists(path) else None


def gen_trainval_test(root, annotated_path, txt_path):
    namelist = os.listdir(os.path.join(root, annotated_path))
    random.shuffle(namelist)
    N = len(namelist)
    print("len: {}".format(N))

    makedir(os.path.join(root, txt_path))
    ftrain = open(os.path.join(root, txt_path, "train.txt"), "w")
    fval = open(os.path.join(root, txt_path, "val.txt"), "w")
    ftest = open(os.path.join(root, txt_path, "test.txt"), "w")

    # 27 : 2 : 1
    n_train = int(N / 30 * 27)
    n_val = int(N / 30 * 2)
    n_trainval = n_train + n_val

    for i in range(N):
        string = "{}\n".format(namelist[i][:-5])
        if i in range(n_train): ftrain.write(string)
        if i in range(n_train, n_trainval): fval.write(string)
        if i in range(n_trainval, N): ftest.write(string)


if __name__ == "__main__":
    root = "./"
    annotated_path = "../annotations"  # json files dir
    txt_path = "ImageSets/Segmentation/"
    gen_trainval_test(root, annotated_path, txt_path)
