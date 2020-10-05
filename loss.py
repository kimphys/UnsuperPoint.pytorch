import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from time import time

import torchvision
import torchvision.transforms as transforms # Apply homography (rotation, scale, skew, perspective transforms)

from args import args
a_usp =args.a_usp
a_pos = args.a_pos
a_scr = args.a_scr
a_xy = args.a_xy
a_desc = args.a_desc
a_decorr = args.a_decorr
lambda_d = args.lambda_d
margin_p = args.margin_p
margin_n = args.margin_n

def ApplyHomography(pA, angle):

    # The angle as the input is written in the degree, NOT RADIAN
    # So, we need to convert it from degree to radian.

    B, C, H, W = pA.shape[0], pA.shape[1], pA.shape[2], pA.shape[3]

    for b in range(B):
        pA_x = pA[b,0,:,:]
        pA_y = pA[b,1,:,:]
        pA[b,0,:,:] = pA_x * torch.cos(angle[b]) - pA_y * torch.sin(angle[b])
        pA[b,1,:,:] = pA_x * torch.sin(angle[b]) + pA_y * torch.cos(angle[b])

    return pA

def PointDistance(pA, pB, angle):

    pA = ApplyHomography(pA, angle)

    x_d = pA[:,0,:,:] - pB[:,0,:,:]
    y_d = pA[:,1,:,:] - pB[:,1,:,:]

    Distance = torch.sqrt(x_d * x_d + y_d * y_d)

    PositionPoint = torch.sum(Distance)

    return Distance, PositionPoint

def PointLoss(angle, a_pos, a_scr, pos_A, pos_B, scr_A, scr_B):

    mse = nn.MSELoss()

    Distance, PositionPoint = PointDistance(pos_A, pos_B, angle)
    ScorePoint = mse(scr_A, scr_B)

    ## Repeatable point
    scr_mean = (scr_A + scr_B) / 2
    RepeatPoint = torch.sum(scr_mean * (Distance - PositionPoint))

    return a_pos * PositionPoint + a_scr * ScorePoint + RepeatPoint

def PointPredLoss(a_xy, pos_r):

    B, C, H, W = pos_r.shape[0], pos_r.shape[1], pos_r.shape[2], pos_r.shape[3]
    
    pos_X = pos_r[:,0,:,:].view(-1, H * W)
    pos_Y = pos_r[:,1,:,:].view(-1, H * W)

    pos_X_sorted, pos_X_indices = torch.sort(pos_X)
    pos_Y_sorted, pos_Y_indices = torch.sort(pos_Y)

    loss_x = torch.pow((pos_X - torch.true_divide((pos_X_indices - 1),(H * W - 1))),2)
    loss_y = torch.pow((pos_Y - torch.true_divide((pos_Y_indices - 1),(H * W - 1))),2)

    return torch.sum(loss_x + loss_y)

def DescLoss(lambda_d, pos_A, pos_B, angle, desc_A, desc_B, margin_p, margin_n):

    B, C, H, W = desc_A.shape[0], desc_A.shape[1], desc_A.shape[2], desc_A.shape[3]
    
    pos_A = ApplyHomography(pos_A, angle)

    pos_A = pos_A.view(B, 2, H * W).transpose(1,2)
    pos_B = pos_B.view(B, 2, H * W).transpose(1,2)
    pos_dist = torch.cdist(pos_A,pos_B)
    
    
    desc_A = desc_A.view(B, C, H * W).transpose(1,2)
    desc_B = desc_B.view(B, C, H * W)

    product = torch.bmm(desc_A, desc_B)

    ones = torch.ones(pos_dist.shape)
    zeros_dist = torch.zeros(pos_dist.shape)   
    zeros_p = torch.zeros(product.shape)  

    if args.gpu is not None:
        ones = ones.cuda(args.gpu, non_blocking=True)
        zeros_dist = zeros_dist.cuda(args.gpu, non_blocking=True)
        zeros_p = zeros_p.cuda(args.gpu, non_blocking=True)

    ConstantMatrix = torch.where(pos_dist >= 8, ones, zeros_dist)
    product_p = torch.where(product < margin_p, product - margin_p, zeros_p)
    product_n = torch.where(product > margin_n, product - margin_n, zeros_p)

    loss = torch.sum(lambda_d * ConstantMatrix * product_p + (1 - ConstantMatrix) * product_n)

    return loss

# Under construction
def DecorreDescLoss(desc_A, desc_B):

    loss = 0

    return loss

def ComputeLoss(angle, scr_A, scr_B, pos_A_r, pos_B_r, pos_A, pos_B, desc_A, desc_B):

    p_loss = PointLoss(angle, a_pos, a_scr, pos_A, pos_B, scr_A, scr_B)

    pred_loss = PointPredLoss(a_xy, pos_A_r) + PointPredLoss(a_xy, pos_B_r)

    d_loss = DescLoss(lambda_d, pos_A, pos_B, angle, desc_A, desc_B, margin_p, margin_n)

    deco_loss = DecorreDescLoss(desc_A, desc_B)

    return a_usp * p_loss + a_xy * pred_loss + a_desc * d_loss + a_decorr * deco_loss                   

if __name__ == "__main__":
    angle = torch.FloatTensor([30, -10])
    scr_A = torch.rand(2,1,32,32)
    scr_B = torch.rand(2,1,32,32)
    pos_A_r = torch.rand(2,2,32,32)
    pos_B_r = torch.rand(2,2,32,32)
    pos_A = torch.rand(2,2,32,32)
    pos_B = torch.rand(2,2,32,32)
    desc_A = torch.rand(2,256,32,32)
    desc_B = torch.rand(2,256,32,32)

    loss = ComputeLoss(angle, scr_A, scr_B, pos_A_r, pos_B_r, pos_A, pos_B, desc_A, desc_B)

    print(loss)