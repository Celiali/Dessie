import matplotlib
matplotlib.use('TkAgg')
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os, torch
import numpy as np
import random
# from easydict import EasyDict as edict

# Functions from: https://github.com/mkocabas/VIBE and T3DP

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def process_image(img, center, scale, points=None, seg = None):
    img, _, _, points_n, seg_n, trans, trans_inv = generate_image_patch(img, center[0], center[1], scale, scale, 256, 256, False, 1.0, 0.0, points, seg)
    #img = img[:, :, ::-1].copy().astype(np.float32)
    img_n = img[:, :, ::-1].copy().astype(np.float32) # RGB
    #return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))
    return np.transpose(img_n, (2, 0, 1)), points_n, seg_n, trans, trans_inv


def generate_image_patch(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot, points = None, seg = None):
    img = cvimg.copy()
    # c = center.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans, trans_inv = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    if seg is not None:
        seg_n = cv2.warpAffine(seg, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)[:,:,0]
    else:
        seg_n = np.zeros(0)

    if points is not None:
        points_n = points.copy()
        for n_jt in range(points.shape[0]):
            points_n[n_jt] = trans_point2d(points_n[n_jt], trans)
    else:
        points_n = np.zeros(0)

    return img_patch, trans, trans_inv, points_n, seg_n, trans, trans_inv


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv

def get_bbox_from_kp2d(kp_2d):
    #new_kp = kp_2d
    zeroindex = np.unique(np.where(kp_2d == 0.)[0])
    #wrongindex = np.where(np.abs(kp_2d[:, 0] - kp_2d[:, 1]) > 1800.)[0]
    #new_kp = np.delete(kp_2d, np.hstack((zeroindex,wrongindex)), 0)
    new_kp = np.delete(kp_2d, zeroindex, 0)

    # get bbox
    #if len(kp_2d.shape) > 2:
    #    ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
    #    lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    #else:
    pad = 0#50
    ul = np.array([new_kp[:, 0].min()-pad, new_kp[:, 1].min()-pad])  # upper left
    lr = np.array([new_kp[:, 0].max()+pad, new_kp[:, 1].max()+pad])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1

    #bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    bbox = np.array([ul[0], ul[1], lr[0], lr[1]])
    return bbox


def transfrom_keypoints(kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):

    scale, rot, do_flip, color_scale = 1.2, 0, False, [1.0, 1.0, 1.0]

    # generate transformation
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale,
        rot,
        inv=False,
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return kp_2d, trans

def get_video_frame(video_file, frame_number):
    import cv2
    videoCapture = cv2.VideoCapture()
    videoCapture.open(video_file)
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, img = videoCapture.read()
    videoCapture.release()
    return img

def get_video_infor(video):
    import cv2
    v = cv2.VideoCapture(video)
    frame_number = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = v.get(cv2.CAP_PROP_FPS)
    v.release()
    return frame_number, width, height, fps

def detect_outlier_1(data):
    datastd = np.std(data, 1, ddof=1, keepdims=True)
    datamean = np.mean(data,1)[:,np.newaxis]
    outlier = np.transpose(np.where(np.abs(data - datamean) > 2*datastd))
    return outlier

def z_score_detect_outlier(data_original,threshold=2.0):
    zeroindex = np.unique(np.where(data_original == 0.)[0])
    data = np.delete(data_original, zeroindex, 0)
    mean_d = np.mean(data)
    std_d = np.std(data)
    outliers = []

    for y in data:
        z_score = (y - mean_d) / std_d
        if np.abs(z_score) > threshold:
            outliers.append(y)
    outlier_index = np.where(np.in1d(data_original,np.array(outliers)))[0]
    return outlier_index

def iqr_detect_outlier(data_original):
    zeroindex = np.unique(np.where(data_original == 0.)[0])
    data = np.delete(data_original, zeroindex, 0)
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    outliers = data[(data > fence_high) | (data < fence_low)]
    outlier_index = np.where(np.in1d(data_original,outliers))[0]
    return outlier_index
