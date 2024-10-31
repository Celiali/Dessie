'''
part of the code adapted from https://github.com/elliottwu/MagicPony/blob/main/magicpony/dataloaders.py
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import argparse
import numpy as np
import torch, random
import logging
from src.train2 import parse_args
from src.test import get_model

from src.render.renderer_pyrender import Renderer
from src.utils.misc import collapseBF
import time
from torch.utils.data import Dataset
import torchvision.datasets.folder
import torchvision.transforms as transforms
import re
from glob import glob

def evaluate_pck(proj_kpts, keypoints, bboxes=None, size=256, INDEX = None):
    '''https://github.com/yufu-wang/aves/blob/main/utils/evaluation.py'''
    PCK10 = []
    error = []

    err = proj_kpts[:, :, :2] - keypoints[:, :, :2]
    err = err.norm(dim=2, keepdim=True)

    if bboxes is not None:
        maxHW, ind = torch.max(bboxes[:, 2:], dim=1)
    else:
        if type(size) == int:
            maxHW = [size] * len(err)
        else:
            maxHW = size

    for i in range(len(err)):
        valid = keypoints[i, :, 2:] > 0
        err_i = err[i][valid]
        error.append(err_i.clone().cpu().data.numpy())
        err_i = err_i / maxHW[i]
        pck10 = (err_i < 0.10).float().mean().item()
        PCK10.append(pck10)

    return PCK10, error

def evaluate_iou(proj_masks, masks):
    '''https://github.com/yufu-wang/aves/blob/main/utils/evaluation.py'''
    IOU = []
    for proj_mask, mask in zip(proj_masks, masks):
        stack = torch.stack([mask, proj_mask]).byte()
        I = torch.all(stack, 0).sum([0, 1]).float()
        U = torch.any(stack, 0).sum([0, 1]).float()
        score = (I / U).item()
        IOU.append(score)

    return IOU

class average_meter():
    '''https://github.com/yufu-wang/aves/blob/main/utils/evaluation.py'''
    def __init__(self, ):
        self.num = 0.
        self.sum = 0.
        self.convert_types = [torch.Tensor, np.ndarray]

    def collect(self, item):
        if type(item) in self.convert_types:
            item = item.flatten()
            item = item.tolist()

        if type(item) is not list:
            item = [item]

        self.sum += sum(item)
        self.num += len(item)

    def report(self, reset=False):
        score = self.sum / self.num
        if reset is True:
            self.reset()

        return score

    def reset(self):
        self.sum = 0.
        self.num = 0.

def none_to_nan(x):
    return torch.FloatTensor([float('nan')]) if x is None else x

def kp_loader_simple(fpath):
    kp = np.load(fpath, allow_pickle=True).item()['keypoints']
    return kp.astype(np.float32)

class RealImageDataset(Dataset):
    def __init__(self, root, in_image_size=256, out_image_size=256, shuffle=False, load_background=False,
                 random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64, load_kp = False, load_bbox = False, ):
        super().__init__()
        self.image_loader = ["rgb.*", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.samples = self._parse_folder(root)
        if shuffle:
            random.shuffle(self.samples)
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size

        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])

        self.image_transform = transforms.Compose( 
            [ transforms.ToTensor(), transforms.Normalize(self.img_mean, self.img_std)])
        self.normalize_transform = transforms.Compose( 
            [ transforms.Normalize(self.img_mean, self.img_std)])
        self.mask_transform = transforms.Compose( 
            [transforms.ToTensor()])
        self.load_background = load_background
        self.random_xflip = random_xflip
        self.load_kp = load_kp
        if load_kp:
            self.kp_loader = ["kp.npy", kp_loader_simple]

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '**/*' + image_path_suffix), recursive=True))
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]
        return result

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.samples)

    def set_random_xflip(self, random_xflip):
        self.random_xflip = random_xflip

    def __getitem__(self, index):
        path = self.samples[index % len(self.samples)]
        orgimages = self._load_ids(path, self.image_loader, transform=self.mask_transform) #[3,256,256]
        masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform)[[0],...] #[1,256,256]
        denormalize_images = orgimages #get_mask_image(orgimages, masks) #[3,256,256] TODO: remove mask
        images = self.normalize_transform(denormalize_images).unsqueeze(0)#[1,3,256,256]
        denormalize_images = denormalize_images.unsqueeze(0)#[1,3,256,256]
        masks = masks.unsqueeze(0)#[1,1,256,256]
        if self.load_kp:
            kp2d = self._load_ids(path, self.kp_loader,transform=torch.FloatTensor).unsqueeze(0) #[1,17,3]
        else:
            kp2d = None
        kp3d, shapeclass, poseclass, textureclass, camera_index_class = None, torch.tensor([0]),torch.tensor([0]), torch.tensor([0]),torch.tensor([[0,0,0,0]])
        
        idx = torch.LongTensor([index])

        out = (*map(none_to_nan, (
            images, masks, denormalize_images, shapeclass, poseclass, textureclass,camera_index_class, kp3d, None, None, idx, kp2d)),)  # for batch collation
        return out

def get_Real_image_loader(data_dir, batch_size=256, num_workers=4, in_image_size=256, out_image_size=256, shuffle=False,
                     load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False,
                     dino_feature_dim=64, load_bbox = False,  load_kp = False):
    dataset = RealImageDataset(data_dir, in_image_size=in_image_size, out_image_size=out_image_size,
                           load_background=load_background, random_xflip=random_xflip,
                           load_dino_feature=load_dino_feature, load_dino_cluster=load_dino_cluster,
                           dino_feature_dim=dino_feature_dim,  load_kp = load_kp, load_bbox = load_bbox)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last = True
    )
    return loader


if __name__ == '__main__':
    args = parse_args()
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt_file))), f"Evaluate_2d_magicpony_{args.ckpt_file.split('/')[8]}_eval.log")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s', 
				    handlers=[
			        logging.FileHandler(save_path, mode='w'),
			        logging.StreamHandler()]
			        )

    # Log arguments
    args_dict = vars(args)
    args_str = ', '.join(f'{key}={value}' for key, value in args_dict.items())
    logging.info(f'Running with arguments: {args_str}')
    logging.info('%s',args.ModelName)
    logging.info('%s',args.ckpt_file)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    Model, trainer = get_model(args, args.ckpt_file, device)
    Model.initial_setup()
    
    faces_cpu = Model.faces_cpu[0]
    renderer = Renderer(focal_length = 5000, center = None, img_w = args.imgsize, img_h = args.imgsize, faces = faces_cpu, same_mesh_color = True)

    get_loader = lambda is_train, random_sample,  **kwargs: get_Real_image_loader(
        batch_size=args.batch_size,
        num_workers=4,
        in_image_size=256,
        out_image_size=args.imgsize,
        load_background='none',
        load_dino_feature=False,
        load_dino_cluster=False,
        dino_feature_dim=16,
        random_xflip='truez' if is_train else False,
        shuffle=random_sample, 
        load_kp = True,
        **kwargs)
    test_loader = get_loader(is_train=False, random_sample=False, data_dir=os.path.join(args.data_dir, 'train'))

    PCK10 = average_meter()
    IOU = average_meter()

    start =time.time()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            pred_data = Model(batch = batch, batch_idx = i)
            pred_vertices = pred_data['pred_vertices_crop'] #[B, 1497,3]
            pred_2d_kp_crop = pred_data['pred_kp2d_crop']   #[B, 17,2]

            gt_mask_crop = collapseBF(batch[1]).squeeze(1)#[B, 1, 256,256]
            gt_kp_3d = None 
            gt_kp_2d = collapseBF(batch[11])#[B, 17,3]
            
            batch_size = pred_data['input_image'].shape[0]
            
            # Render and save all results
            proj_masks = []
            obj_sizes = []
            for j in range(batch_size):
                depth = renderer(pred_vertices[j].detach().cpu().numpy(), obtainSil = True) #[256,256]
                proj_masks.append(torch.tensor(depth))

                mask = gt_mask_crop[j].cpu()
                ind = torch.nonzero(mask > 1e-3)
                if ind.shape[0] == 0:
                    obj_sizes.append(args.imgsize)  
                else:  
                    h = ind[:, 0].max() - ind[:, 0].min()
                    w = ind[:, 1].max() - ind[:, 1].min()
                    obj_sizes.append(max(h, w))

            # Evaluate fitting quality
            proj_masks = torch.stack(proj_masks)
            iou = evaluate_iou(proj_masks, gt_mask_crop.cpu())
            
            kpts_gt = gt_kp_2d.float().cpu()
            kpts_2d = pred_2d_kp_crop.cpu() 
            pck10, kperror = evaluate_pck(kpts_2d, kpts_gt, size=obj_sizes)    
                
            PCK10.collect(pck10)
            IOU.collect(iou)
                
    logging.info('%4f', time.time() - start)
    logging.info('%s %.4f', 'Over all PCK10:', PCK10.report())
    logging.info('%s %.4f', 'Over all IOU:', IOU.report())