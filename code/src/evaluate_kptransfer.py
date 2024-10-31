import sys,os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
import os
import argparse
import numpy as np
import torch, random
import logging
from src.train2 import parse_args
from src.test import get_model
import os.path as osp
from src.evalkp_utils.lassierender import Renderer
import torchvision
from src.evalkp_utils.visualize_utils import *
from src.evalkp_utils.data_utils import *
import json
from torch.utils.data import Dataset
from src.evalkp_utils.config import cfg

class ImageDataset(Dataset):
    def __init__(self, REALPATH):
        anno_file = os.path.join(REALPATH, f'pascal_val/annotations/pascal_val.jsons')
        with open(anno_file) as f:
            anno = json.load(f)
        # load filelist
        pascal_filelist = sorted([str(ann['img_id']) for ann in anno])
        num_imgs = len(pascal_filelist)
        print('pascal: {} valid images'.format(num_imgs))
        for ann in anno:
            ann['image_path'] = os.path.join(REALPATH, f'pascal_val/images', os.path.basename(ann['image_path']) )
        self.filelist = pascal_filelist
        self.anno = {str(an['img_id']): an for an in anno}
        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.norm_transform = transforms.Normalize(img_mean, img_std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_id = self.filelist[idx]
        img_path = self.anno[str(img_id)]['image_path']
        img = cv2.imread(img_path) / 255.
        img = img[:, :, ::-1]
        denormalized_tensor = torch.from_numpy(img.copy()).permute(2,0,1)[None,:,:,:].float()
        img_tensor = self.norm_transform(denormalized_tensor)
        joints = torch.from_numpy(np.array(self.anno[img_id]['kp'])[None,:,:]) 
        out  = (img_tensor, torch.FloatTensor([[0]]), denormalized_tensor,torch.FloatTensor([[0]]), torch.FloatTensor([[0]]), torch.FloatTensor([[0]]),
            torch.FloatTensor([[0]]), torch.FloatTensor([[0]]),torch.FloatTensor([[0]]),torch.FloatTensor([[0]]),torch.FloatTensor([[0]]),
            joints)
        return out

if __name__ == '__main__':
    args = parse_args()
    cfg.set_args("horse")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("========== Loading data... ========== ")
    # get the dataset
    dataset = ImageDataset(args.REALPATH)
    print(len(dataset))
    # get the data loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle = False, num_workers=1)
    num_imgs = len(test_loader)
    assert args.batch_size == 1, 'batch size should be 1'
    assert args.data_batch_size == 1, 'batch size should be 1'
    print("========== Preparing model... ========== ")
    Model, trainer = get_model(args, args.ckpt_file, device)
    Model.initial_setup()
    Model.eval()
    save_path = osp.join(osp.dirname(osp.dirname(args.ckpt_file)),'kp_tranfer')
    if not osp.exists(save_path):
        os.makedirs(save_path)
    print("========== Initial render... ========== ")
    CIRender = Renderer(device, 'text')
    rasterizer = CIRender.renderer.rasterizer
    faces = Model.faces

    print("========== Predict data... ========== ")
    outputs = {'verts':[], 'verts_2d': []}
    inputs = {'kps_gt':[], 'part_masks':[]}
    save_img = [] # save for further visualization
    original_img = [] # save for further visualization
    for i, batch in enumerate(test_loader):
        pred_data = Model(batch = batch, batch_idx = 0)
        verts_3d = pred_data['pred_vertices_crop']
        screen_size = torch.tensor((args.imgsize,args.imgsize)).unsqueeze(0).to(cfg.device)
        verts_2d = Model.render.camera.transform_points_screen(verts_3d, image_size=screen_size)[:, :, :2] 
        outputs['verts'].append(verts_3d)
        outputs['verts_2d'].append(verts_2d)
        inputs['kps_gt'].append(batch[-1][0]) 
        inputs['part_masks'].append(batch[1][0])

        img = pred_data['pred_color_img'].cpu().numpy().transpose(0, 2, 3, 1)
        img = torchvision.transforms.ToPILImage()((img[0]*255).astype(np.uint8))
        save_img.append(img)
        original_img.append(batch[2][0].cpu().numpy())

    print("========== Keypoint transfer evaluation... ========== ")
    num_pairs = 0
    pck05 = 0
    pck10 = 0
    for i1 in range(num_imgs):
        for i2 in range(num_imgs):
            if i1 == i2:
                continue
            kps1 = inputs['kps_gt'][i1][0].cpu() 
            kps2 = inputs['kps_gt'][i2][0].cpu() 
            verts1 = outputs['verts_2d'][i1].cpu().reshape(-1,2) 
            verts2 = outputs['verts_2d'][i2].cpu().reshape(-1,2) 
            verts1_vis = CIRender.get_verts_vis(outputs['verts'][i1], faces).cpu()
            v_matched = find_nearest_vertex(kps1, verts1, verts1_vis)
            kps_trans = verts2[v_matched]
            valid = (kps1[:,2] > 0) * (kps2[:,2] > 0)
            if valid.sum() == 0:
                continue
            kps_trans_cal = kps_trans[:,:2]/args.imgsize
            kps2_cal = kps2[:,:2]/args.imgsize
            dist = ((kps_trans_cal - kps2_cal[:,:2])**2).sum(1).sqrt()
            pck05 += ((dist <= 0.05) * valid).sum() / valid.sum()
            pck10 += ((dist <= 0.1) * valid).sum() / valid.sum()
            num_pairs += 1

            # ###### visualize
            if i1 < 10 and i2<10 :
                source_img = original_img[i1][0]
                target_img = original_img[i2][0]
                source_verts_array = (verts1*verts1_vis[:,None]).cpu().data.numpy() # [1497,2]
                source_kp_array = verts1[v_matched].cpu().data.numpy()
                source_visible = (kps1[:,2]>0).cpu().data.numpy()
                source_gt_array = (kps1[:,:2]).cpu().data.numpy()
                target_kps_array = kps_trans.cpu().data.numpy()
                target_gt_array = kps2[:,:2].cpu().data.numpy()
                target_visible = (kps2[:,2]>0).cpu().data.numpy()
                kps_err = dist.cpu().data.numpy()
                visible = valid.cpu().data.numpy()
                source_pred_img = np.array(save_img[i1])[:,256:,:]
                target_pred_img = np.array(save_img[i2])[:,256:,:]
                visualize_image(source_img, target_img, source_verts_array,source_kp_array,source_visible,source_gt_array,
                        target_kps_array,target_gt_array,target_visible,visible,kps_err,save_path = save_path,flag = f'{i1}_{i2}',
                        source_pred_img = source_pred_img, target_pred_img = target_pred_img)
            
    pck05 /= num_pairs
    pck10 /= num_pairs
    print('PCK05=%.4f' % pck05)
    print('PCK10=%.4f' % pck10)

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt_file))), f"KPtransfer_{args.ckpt_file.split('/')[8]}_eval.log")
    with open(save_path,'w') as f:
        # Log arguments
        args_dict = vars(args)
        args_str = ', '.join(f'{key}={value}' for key, value in args_dict.items())
        f.write(f'Arguments: {args_str}\n')
        f.write('=>pck05: %.4f\n' % pck05)
        f.write('=>pck10: %.4f\n' % pck10)
    f.close()