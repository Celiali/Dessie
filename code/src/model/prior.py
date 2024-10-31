import pickle as pkl
import numpy as np
from chumpy import Ch
# from body.matlab import row
#import cv2
from src.utils.geometry import rotmat_to_axis_angle
import torch

class Poseprior(object):
    def __init__(self, prior_path, device):
        with open(prior_path, "rb") as f:
            res = pkl.load(f, encoding='latin1')

        self.mean_ch = res['mean_pose']
        self.precs_ch = res['pic']

        self.precs = torch.from_numpy(res['pic'].r.copy()).float().to(device)
        self.mean = torch.from_numpy(res['mean_pose']).float().to(device)

        # Ignore the first 3 global rotation.
        prefix = 3
        pose_len = 108

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False

        self.use_ind_tch = torch.from_numpy(self.use_ind).float().to(device)

    def __call__(self, x):
        # remove the sequence length
        if len(x.shape) == 5:
            # B, T, 36, 3, 3 --> B, 36,3,3
            batch_size_all = x.shape[0] * x.shape[1]
            x = x.view(batch_size_all, 36, 3, 3)
        elif len(x.shape) == 3:
            # B, T, 108 --> B, 108
            batch_size_all = x.shape[0] * x.shape[1]
            x = x.view(batch_size_all, -1)

        if len(x.shape) == 4:
            assert x.shape[1:] == (36,3,3)
            x =rotmat_to_axis_angle(x)
        mean_sub = x.reshape(-1, 36 * 3)[:, self.use_ind] - self.mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, self.precs, dims=([1], [0])) 
        return res ** 2
