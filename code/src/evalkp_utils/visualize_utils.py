'''code adapted from https://github.com/elliottwu/MagicPony/tree/main '''

import logging
import random
import tarfile
from pathlib import Path

import configargparse
import cv2,os
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

def visualize_image(
    source_img,
    target_img,
    source_verts_array,
    source_kp_array,
    source_visible,
    source_gt_array,
    target_kps_array,
    target_gt_array,
    target_visible,
    visible,
    kps_err,
    save_path,
    flag,
    source_pred_img, target_pred_img
):
    # source image
    # load image
    source_img = (source_img*255).transpose(1,2,0) #Image.open(source_img_path)

    # visualize keypoints
    source_kp_plot = plot_points(source_kp_array[:, :2], source_img.copy(), visible=source_visible)
    source_gt_plot = plot_points(source_gt_array[:, :2], source_img.copy(), visible=source_visible)
    # load verts visuals
    source_verts_plot = visualize_vertices(source_verts_array, np.array(source_img))

    # target image
    # load image
    target_img = (target_img*255).transpose(1,2,0) # Image.open(target_img_path)
    # load verts visuals
    target_kp_plot = plot_points(target_kps_array[:, :2], target_img.copy(), visible=visible)
    target_gt_plot = plot_points(target_gt_array[:, :2], target_img.copy(), visible=target_visible)

    pck1 = ((kps_err < 0.1) * visible).sum() / visible.sum()
    target_kp_pred_plot = cv2.putText(
        target_kp_plot.copy(),
        f"pck@0.1: {pck1:0.4f}",
        (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    vis = arrange(
        [
            [source_gt_plot, source_kp_plot, source_verts_plot,source_pred_img],
            [target_gt_plot, target_kp_plot, target_kp_pred_plot,target_pred_img],
        ]
    )
    Image.fromarray(vis.astype(np.uint8)).save(os.path.join(save_path, f'{flag}.jpg'))


# get "maximally" different random colors:
#  ref: https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor=0.5, seed=None):
    if seed is None:
        seed = random.randint()
    r = random.Random(seed)
    return [
        (x + pastel_factor) / (1.0 + pastel_factor)
        for x in [r.uniform(0, 1.0) for i in [1, 2, 3]]
    ]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor, seed=i)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for _ in range(n):
        colors.append(generate_new_color(colors, pastel_factor=pastel_factor))
    return colors


def plot_points(points, image, visible=None, correct=None):
    colors = get_n_colors(len(points), pastel_factor=0.2)
    if correct is None:
        correct = [1] * len(points)
    for i, (coord, color, visible_, correct_) in enumerate(
        zip(points, colors, visible, correct)
    ):
        if visible_ == 1:
            color = [255 * c for c in color]
            if correct_:
                image = cv2.circle(
                    np.ascontiguousarray(image),
                    tuple(coord.astype("int32")),
                    4,
                    color,
                    2,
                )
            else:
                # plot x
                image = cv2.line(
                    np.ascontiguousarray(image),
                    tuple(coord.astype("int32") - 4),
                    tuple(coord.astype("int32") + 4),
                    color,
                    2,
                )
                image = cv2.line(
                    np.ascontiguousarray(image),
                    tuple(coord.astype("int32") + np.array([-4, 4])),
                    tuple(coord.astype("int32") - np.array([-4, 4])),
                    color,
                    2,
                )
            # plot index next to point
            image = cv2.putText(
                np.ascontiguousarray(image),
                str(i),
                tuple(coord.astype("int32") + np.array([4, 4])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return image


def visualize_vertices(verts, img, cmap="hot"):
    h, w = img.shape[:2]
    # verts = (verts + 1) / 2 * np.array([w, h])
    verts = np.round(verts).astype("int32")
    img = 0.5 * img + 0.5 * 255
    cmap = plt.cm.get_cmap(cmap)
    for i, v in enumerate(verts):
        x, y = v
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        img[y, x] = np.array(cmap(i / len(verts)))[:3] * 255
    return img.astype("uint8")


def arrange(images):
    rows = []
    for row in images:
        rows += [np.concatenate(row, axis=1)]
    image = np.concatenate(rows, axis=0)
    return image