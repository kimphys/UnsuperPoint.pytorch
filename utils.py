from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

def merge(imgs, row):
    B, C, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]

    q, r = divmod(B, row)

    if not r == 0:
        new_image = Image.new("RGB", (W * row, (q + 1) * H), (256,256,256))
    else:
        new_image = Image.new("RGB", (W * row, q * H), (256,256,256))
    trans = torchvision.transforms.ToPILImage()

    i = 0
    j = 0

    for b in range(B):

        if not i < row:
            i = 0
            j = j + 1
        
        leftupper = (i * W, j * H)
        img = trans(imgs[b])
        new_image.paste(img, leftupper)
        
        i = i + 1

    return new_image

def save_train_samples(imgs_A, imgs_B, idx, row=3):

    imgs_A_sample = merge(imgs_A, row)
    imgs_B_sample = merge(imgs_B, row)

    imgs_A_sample.save("{}_A.png".format(idx))
    imgs_B_sample.save("{}_B.png".format(idx))

    return