"""
code adapted from https://github.com/statho/animals3d/blob/main/acsm/acsm/datasets/base.py
"""

from __future__ import absolute_import, division, print_function
import cv2
import imageio
import numpy as np
from absl import flags
from torch.utils.data import Dataset
import torchvision

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.staths.constants as constants
import src.staths.image_utils as image_utils
import src.staths.transformations as transformations


class BaseDataset(Dataset):
    '''
    Base data loading class.
    Returns:
        - img : B X 3 X H X W
        - joints  : B X num_joints X 2
        - mask (optional) : B X H X W
        - sfm_pose (optional): B X 7 (s, tr, q)
        (joints, sfm_pose) correspond to image coordinates in [-1, 1]
    '''
    def __init__(self, opts):
        self.opts         = opts
        self.img_size     = opts.img_size
        self.jitter_frac  = opts.jitter_frac
        self.padding_frac = opts.padding_frac

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        # read image
        img_id   = self.filelist[index]
        img_path = self.anno_dict[img_id]['img_path']
        img = imageio.imread(img_path) / 255.0
        if len(img.shape) == 2: # for grayscale images
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

        # get bbox ([x0, y0, width, height])
        bbox    = self.anno_dict[img_id]['img_bbox']
        bbox[2]+= bbox[0]
        bbox[3]+= bbox[1]

        # get joints
        joints = np.array(self.anno_dict[img_id]['joints']) if self._out_joints else None

        # get mask (optionally)
        mask = np.expand_dims(self.anno_dict[img_id]['seg'], 2) if self._out_mask else None

        # get sfm camera (optionally) -- used only during evaluation
        if self._out_pose:
            scale = np.copy(self.anno_dict[img_id]['cam_scale'])
            trans = np.copy(self.anno_dict[img_id]['cam_trans'])
            rot   = np.copy(self.anno_dict[img_id]['cam_rot'])
            sfm_pose      = [scale, trans, rot]
            sfm_rot       = np.pad(sfm_pose[2], (0, 1), 'constant')
            sfm_rot[3, 3] = 1
            sfm_pose[2]   = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)
        else:
            sfm_pose = None

        # -------------

        ## Data transformations
        if self.opts.tight_crop:
            self.padding_frac = 0.0

        # peturb and create square bbox
        jitter_frac = self.jitter_frac if self.opts.split=='train' else 0
        bbox = image_utils.peturb_bbox(bbox, pf=self.padding_frac, jf=jitter_frac)

        if not self.opts.tight_crop:
            bbox = image_utils.square_bbox(bbox)

        # crop image (and optionally mask) around bbox, translate joints
        img, mask, joints, sfm_pose = self.crop_image(img, bbox, mask=mask, joints=joints, sfm_pose=sfm_pose)

        # scale image (and optionally mask), and scale joints accordingly
        if self.opts.tight_crop:
            img, mask, joints, sfm_pose = self.scale_image_tight(img, mask=mask, joints=joints, sfm_pose=sfm_pose)
        else:
            img, mask, joints, sfm_pose = self.scale_image(img, mask=mask, joints=joints, sfm_pose=sfm_pose)

        # mirror image on random
        if self.opts.split == 'train':
            img, mask, joints, sfm_pose = self.mirror_image(img, mask=mask, joints=joints, sfm_pose=sfm_pose)

        img_h, img_w = img.shape[:2]
        if self._out_joints:
            joints = self.normalize_joints(joints, img_h, img_w) # normalize joints to be [-1, 1]
        if self._out_pose:
            sfm_pose = self.normalize_pose(sfm_pose, img_h, img_w)

        # transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        elem  = { 'img_name': img_id, 'inds': index, 'img': img }
        if self._out_joints:
            elem['kp'] = joints
        if self._out_mask:
            elem['mask'] = mask
        if self._out_pose:
            elem['sfm_pose']   = sfm_pose
            elem['valid_cam']  = self.anno_dict[img_id]['valid_cam']
        elem['img_h'] = img_h
        elem['img_w'] = img_w

        return elem


    def normalize_joints(self, joints, img_h, img_w):
        vis    = joints[:, 2, None] > 0
        new_joints = np.stack( [2 * (joints[:, 0] / img_w) - 1, 2 * (joints[:, 1] / img_h) - 1, joints[:, 2]] ).T
        new_joints = vis * new_joints
        return new_joints

    def normalize_pose(self, sfm_pose, img_h, img_w):
        sfm_pose[0]   *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        return sfm_pose

    def crop_image(self, img, bbox, mask=None, joints=None, sfm_pose=None):
        img  = image_utils.crop(img, bbox, bgval=1)
        if mask is not None:
            mask = image_utils.crop(mask, bbox, bgval=0)
        if joints is not None:
            vis = joints[:, 2] > 0
            joints[vis, 0] -= bbox[0]
            joints[vis, 1] -= bbox[1]
        if sfm_pose is not None:
            sfm_pose[1][0] -= bbox[0]
            sfm_pose[1][1] -= bbox[1]
        return img, mask, joints, sfm_pose

    def scale_image(self, img, mask=None, joints=None, sfm_pose=None):
        bwidth  = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale   = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)

        if mask is not None:
            mask, _ = image_utils.resize_img(mask, scale)
        if joints is not None:
            vis = joints[:, 2] > 0
            joints[vis, :2] *= scale
        if sfm_pose is not None:
            sfm_pose[0] *= scale
            sfm_pose[1] *= scale

        return img_scale, mask, joints, sfm_pose

    def scale_image_tight(self, img, mask=None, joints=None, sfm_pose=None):
        bwidth  = np.shape(img)[1]
        bheight = np.shape(img)[0]
        scale_x = self.img_size/bwidth
        scale_y = self.img_size/bheight
        img_scale = cv2.resize(img, (self.img_size, self.img_size))

        if mask is not None:
            mask_scale = cv2.resize(mask, (self.img_size, self.img_size))
        if joints is not None:
            vis = joints[:, 2] > 0
            joints[vis, 0:1] *= scale_x
            joints[vis, 1:2] *= scale_y
        if sfm_pose is not None:
            sfm_pose[0] *= scale_x
            sfm_pose[1] *= scale_y

        return img_scale, mask_scale, joints, sfm_pose

    def mirror_image(self, img, mask=None, joints=None, sfm_pose=None):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            if mask is not None:
                mask = mask[:, ::-1].copy()
            if joints is not None:
                # Flip joints
                new_x = img.shape[1] - joints[:, 0] - 1
                joints_flip = np.hstack((new_x[:, None], joints[:, 1:]))
                joints_flip = joints_flip[constants.QUAD_JOINT_PERM, :]
                joints = joints_flip
            if sfm_pose is not None:
                # Flip sfm_pose Rot
                R = transformations.quaternion_matrix(sfm_pose[2])
                flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
                sfm_pose[2] = transformations.quaternion_from_matrix( flip_R, isprecise=True )
                tx = img.shape[1] - sfm_pose[1][0] - 1  # flip tx
                sfm_pose[1][0] = tx
            return img_flip, mask, joints, sfm_pose
        else:
            return img, mask, joints, sfm_pose
