import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from functools import reduce

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def kp2d_loss(projected_keypoints, keypoints_2d):

    keypoints_conf = keypoints_2d[:, :, -1]
    threshold = nn.Threshold(0.5, 0.)
    # threshold = nn.Threshold(0.7, 0.)
    keypoints_conf_new = threshold(keypoints_conf)

    # Weighted robust reprojection loss
    sigma = 50
    reprojection_error = gmof(projected_keypoints - keypoints_2d[:,:,:2], sigma)
    reprojection_loss = (keypoints_conf_new ** 2) * reprojection_error.sum(dim=-1)
    kp_loss = reprojection_loss.sum() / (keypoints_conf_new ** 2).sum()
    return kp_loss


def betas_loss(betas_pred, betas_gt=None, prec=None):
    """
    betas_pred: ( B ,9 )
    """
    if len(betas_pred.shape) == 3:
        # B, T, 9
        batch_size_all = betas_pred.shape[0]* betas_pred.shape[1]
        betas_pred = betas_pred.view(batch_size_all, -1)
    if betas_gt is None:
        if prec is None:
            b_loss = betas_pred ** 2
            return b_loss.mean()
        else:
            b_loss = betas_pred * betas_pred  # *prec
            return b_loss.mean()
    else:
        criterion = torch.nn.MSELoss()
        return criterion(betas_pred, betas_gt)

def gt_loss(pred, gt):
    criterion = torch.nn.MSELoss()
    return criterion(pred, gt).mean()

def mask_loss(proj_masks, masks):
    # L1 mask loss
    total_loss = F.smooth_l1_loss(proj_masks, masks, reduction='none').sum(dim=[1, 2])
    batch_size = total_loss.shape[0]
    return total_loss.sum()/batch_size   

def latentL2_loss(data, index):
    '''
    :param data: # [N, 2, 512]
    :param labels: # [N, 1]
    :return:
    '''
    if index.sum() == 0:
        loss = torch.tensor(0.0, device=data.device)
    else:
        if len(index.shape) != 1:
            index = index.reshape(-1)
        loss = F.mse_loss(data[index,0,:], data[index,1,:])
    return loss