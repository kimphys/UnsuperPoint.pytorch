import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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

    angle_r = angle * math.pi / 180

    B, C, H, W = pA.shape[0], pA.shape[1], pA.shape[2], pA.shape[3]

    for b in range(B):
        for i in range(H): 
            for j in range(W):
                pA_x = pA[b,0,i,j]
                pA_y = pA[b,1,i,j]
                
                pA[b,0,i,j] = pA_x * math.cos(angle_r[b]) - pA_y * math.sin(angle_r[b])
                pA[b,1,i,j] = pA_x * math.sin(angle_r[b]) + pA_y * math.cos(angle_r[b])

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
    RepeatPoint = scr_mean * (Distance - PositionPoint)
    RepeatPoint = torch.sum(RepeatPoint)

    return a_pos * PositionPoint + a_scr * ScorePoint + RepeatPoint

def DistributionDistance(pos_sorted, M):

    DDistance = 0

    for i in range(M):
        DDistance += (pos_sorted[i] - (i - 1)/(M - 1)) * (pos_sorted[i] - (i - 1)/(M - 1))

    return DDistance

def PointPredLoss(a_xy, pos_r):

    H, W = pos_r.shape[2], pos_r.shape[3]
    
    pos_X = pos_r[:,0,:,:].view(-1, H * W)
    pos_Y = pos_r[:,1,:,:].view(-1, H * W)

    pos_X_sorted, pos_X_indices = torch.sort(pos_X)
    pos_Y_sorted, pos_Y_indices = torch.sort(pos_Y)

    loss_x = DistributionDistance(pos_X_sorted[0], H * W)
    loss_y = DistributionDistance(pos_Y_sorted[0], H * W)

    return loss_x + loss_y

def CorrespondMatrix(pos_A, pos_B, angle):

    B, C, H, W = pos_A.shape[0], pos_A.shape[1], pos_A.shape[2], pos_A.shape[3]
    pos_A = ApplyHomography(pos_A, angle)

    pos_A_x = pos_A[:,0,:,:].view(-1, H * W)
    pos_A_y = pos_A[:,1,:,:].view(-1, H * W)

    pos_B_x = pos_B[:,0,:,:].view(-1, H * W)
    pos_B_y = pos_B[:,1,:,:].view(-1, H * W)

    Correspondence = torch.zeros(B, H * W, H * W)

    for i in range(H * W):
        for j in range(H * W):
            distance = (pos_A_x[:,i] - pos_B_x[:,j]).pow(2)
            distance += (pos_A_y[:,i] - pos_B_y[:,j]).pow(2)
            distance = torch.sqrt(distance)
            Correspondence[:,i,j] = distance

    return Correspondence

def DescLoss(lambda_d, pos_A, pos_B, angle, desc_A, desc_B, margin_p, margin_n):

    B, C, H, W = desc_A.shape[0], desc_A.shape[1], desc_A.shape[2], desc_A.shape[3]
                
    Correspondence = CorrespondMatrix(pos_A, pos_B, angle)

    loss = 0
    desc_A = desc_A.view(B,C,H * W)
    desc_B = desc_B.view(B,C,H * W)

    for b in range(B):
        for i in range(H * W):
            for j in range(H * W):
                constant = 1 if Correspondence[b,i,j] >= 8 else 0

                f_A = desc_A[b,:,i]
                f_B = desc_B[b,:,j]

                f_product = torch.dot(f_A,f_B)

                loss += lambda_d * constant * max(0,margin_p - f_product)
                loss += (1 - constant) * max(0, f_product - margin_n)

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
    scr_A = torch.rand(2,1,20,20)
    scr_B = torch.rand(2,1,20,20)
    pos_A_r = torch.rand(2,2,20,20)
    pos_B_r = torch.rand(2,2,20,20)
    pos_A = torch.rand(2,2,20,20)
    pos_B = torch.rand(2,2,20,20)
    desc_A = torch.rand(2,256,20,20)
    desc_B = torch.rand(2,256,20,20)

    loss = ComputeLoss(angle, scr_A, scr_B, pos_A_r, pos_B_r, pos_A, pos_B, desc_A, desc_B)

    print(loss)