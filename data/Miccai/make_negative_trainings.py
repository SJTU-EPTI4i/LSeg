import cv2
from PIL import Image
import numpy as np
import os
from shutil import copyfile

# 生成阴性样本
def make_negative(img, mask1, mask2):
    timg = cv2.imread(img)
    # 去掉图片中被mask的部分

    # 处理IrMA
    if mask1 != None:
        tmask1 = cv2.imread(mask1, cv2.COLOR_BGR2GRAY)
        tmask1 = cv2.bitwise_not(tmask1, tmask1)
        timg1 = cv2.bitwise_and(timg, timg, mask=tmask1)
        cv2.imwrite(img[:-4] + "_m1.png", timg1)
        if mask2 != None:
            copyfile(mask2, mask2[:-4] + "_m1.png")

    # 处理Neovascularization
    if mask2 != None:
        tmask2 = cv2.imread(mask2, cv2.COLOR_BGR2GRAY)
        tmask2 = cv2.bitwise_not(tmask2, tmask2)
        timg2 = cv2.bitwise_and(timg, timg, mask=tmask2)
        cv2.imwrite(img[:-4] + "_m2.png", timg2)
        if mask1 != None:
            copyfile(mask1, mask1[:-4] + "_m2.png")

    # 如果两者都有，再叠加去掉
    if mask1 != None and mask2 != None:
        tmask3 = cv2.bitwise_and(tmask1, tmask2)
        timg3 = cv2.bitwise_and(timg, timg, mask=tmask3)
        cv2.imwrite(img[:-4] + "_m3.png", timg3)

path_train = "A. Segmentation/1. Original Images/a. Training Set/"
path_test1 = "A. Segmentation/2. Groundtruths/a. Training Set/1. Intraretinal Microvascular Abnormalities/"
path_test3 = "A. Segmentation/2. Groundtruths/a. Training Set/3. Neovascularization/"

files = os.listdir(path_train)
for f in files:
    mask1 = path_test1 + f if os.path.exists(path_test1 + f) else None
    mask2 = path_test3 + f if os.path.exists(path_test3 + f) else None
    make_negative(path_train + f, mask1, mask2)

