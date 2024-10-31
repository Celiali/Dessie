import os
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import torchvision.utils as tvutils
import zipfile
from PIL import Image
from einops import rearrange

def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')

def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)

def image_grid(tensor, nrow=None):
    b, c, h, w = tensor.shape
    if nrow is None:
        nrow = int(np.ceil(b**0.5))
    if c == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tvutils.make_grid(tensor, nrow=nrow, normalize=False)
    return tensor

def video_grid(tensor, nrow=None):
    return torch.stack([image_grid(t, nrow=nrow) for t in tensor.unbind(1)], 0)

def validate_tensor_to_device(x, device):
    if torch.any(torch.isnan(x)):
        return None
    else:
        return x.to(device)
    
def validate_tensor(x):
    if torch.any(torch.isnan(x)):
        return None
    else:
        return x

def resize_and_pad(img, target_width, target_height, fill_color=(255, 255, 255)):
    """
    Resize the image to fit within the target dimensions, and then pad it to the exact target dimensions.
    """
    # Calculate aspect ratio
    aspect_ratio = img.width / img.height
    new_width = target_width
    new_height = int(new_width / aspect_ratio)

    if new_height > target_height:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new blank image with the target dimensions
    new_img = Image.new("RGB", (target_width, target_height), fill_color)

    # Paste the resized image onto the center of the blank image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    new_img.paste(img, (x_offset, y_offset))

    return new_img

def get_next_version(save_dir):
    if not os.path.exists(save_dir):
        return 0
    listdir_info = os.listdir(save_dir)
    existing_versions = []
    if len(listdir_info ) ==0:
        return 0
    for bn in listdir_info:
        if bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1