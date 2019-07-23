# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.19
    @readme: The implementation of PRNet Network Loss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

import cv2
import numpy as np


def preprocess(mask):
    """
    :param mask: grayscale of mask.
    :return:
    """
    tmp = {}
    mask[mask > 0] = mask[mask > 0] / 16
    mask[mask == 15] = 16
    mask[mask == 7] = 8
    # for i in mask:
    #     for j in i:
    #         if j not in tmp.keys():
    #             tmp[j] = 1
    #         else:
    #             tmp[j] += 1
    # print(tmp)
    # {0: 21669, 3: 33223, 4: 10429, 8: 147, 16: 68}

    return mask


class WeightMaskLoss(nn.Module):
    """
        L2_Loss * Weight Mask
    """

    def __init__(self, mask_path):
        super(WeightMaskLoss, self).__init__()
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, 0)
            self.mask = torch.from_numpy(preprocess(self.mask)).float().to("cuda")
        else:
            raise FileNotFoundError("Mask File Not Found! Please Check your Settings!")

    def forward(self, pred, gt):
        result = torch.mean(torch.pow((pred - gt), 2), dim=1)
        result = torch.mul(result, self.mask)

        # 1) 官方(不除256*256的话, 数值就太大了...).
        result = torch.sum(result)
        result = result / (self.mask.size(1) ** 2)
        # 2) 一般使用的都是mean.
        # result = torch.mean(result)
        return result


def INFO(*inputs):
    if len(inputs) == 1:
        print("[ PRNet ] {}".format(inputs))
    elif len(inputs) == 2:
        print("[ PRNet ] {0}: {1}".format(inputs[0], inputs[1]))


if __name__ == "__main__":
    # mask = cv2.imread("/home/samuel/gaodaiheng/3DFace/code/PRNet_Samuel/utils/uv_data/uv_weight_mask_gdh.png", 0)
    # preprocess(mask)
    INFO("Random Seed", 1)