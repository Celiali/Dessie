import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import reduce
import numpy as np
import os
from src.render.render_p3d import base_renderer
from src.SMAL.smal_torch.smal_torch import SMAL
from src.model.prior import Poseprior
from src.model.dinohmr import DINOHMR
from src.model.dessie import DESSIE
from src.model.loss import *
from src.utils.model_utils import get_point,organize_kp
from src.utils.misc import *
from src.utils.geometry import batch_rodrigues
import subprocess
import re

class VanillaUpdateRealImageTrainer(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        print("trainer: VanillaUpdateRealImageTrainer")
        # set trainable model
        if opts.ModelName == 'DINOHMR':
            self.model = DINOHMR(opts)
        elif opts.ModelName == 'DESSIE':
            self.model = DESSIE(opts)
        else:
            raise NotImplementedError

        self.HMRKEYS = ['DINOHMR']
        self.DISENKEYS = [ 'DESSIE']
        # set initial learning rate
        self.lr = abs(opts.lr)

        self.opts = opts
        self.img_size = np.array([opts.imgsize, opts.imgsize])
        self.FOCAL_LENGTH_BBOX = 5000
        cam_rot = torch.eye(3).unsqueeze(0)
        cam_t = torch.zeros(3).unsqueeze(0)
        self.register_buffer('cam_rot', cam_rot)
        self.register_buffer('cam_t', cam_t)

        self.initial = False

    def setup(self, stage=None):
        # Initialization code that doesn't depend on the device
        pass

    def initial_setup(self):  # on_fit_start(self): #on_pretrain_routine_start
        # Code that should run before training or validation
        if not self.initial:
            # print("on_fit_start")
            self.prior = Poseprior(
                prior_path=os.path.join(self.opts.model_dir,
                                        'walking_toy_symmetric_smal_0000_new_skeleton_pose_prior_new_36parts.pkl'),
                device=self.device)
            self.render = base_renderer(size=self.img_size[0], focal=self.FOCAL_LENGTH_BBOX, device=self.device,
                                        colorRender=True)
            # setup SMAL skinning
            self.smal_model = SMAL(os.path.join(self.opts.model_dir, 'my_smpl_0000_horse_new_skeleton_horse.pkl'),
                                   device=self.device)
            self.faces_cpu = self.smal_model.faces.unsqueeze(0).cpu().data.numpy()
            self.faces = self.smal_model.faces.unsqueeze(0)
            self.initial = True
            self.image_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            self.image_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def cam_bbox(self, crop_cam, r=256, focal_length=5000.):
        bs = r * crop_cam[:, 0] + 1e-9
        tz = 2 * focal_length / bs
        tx = crop_cam[:, 1]
        ty = crop_cam[:, 2]
        full_cam = torch.stack([tx, ty, tz], dim=-1)
        return full_cam

    def get_SMAL_results(self, betas, poses, trans):
        # input (B*T,x)
        # output (B*T,x)
        batch_size = poses.shape[0]
        type = poses.dtype

        if trans is None:
            default_transl = torch.zeros([batch_size, 3],
                                         dtype=type,
                                         requires_grad=True).to(self.device)
        else:
            default_transl = trans
        if betas is None:
            betas = torch.zeros([batch_size, 9],
                                dtype=type,
                                requires_grad=True).to(self.device)
        verts, joints, Rs = self.smal_model(betas, poses, default_transl)
        return verts, joints, Rs

    def get_render_results(self, verts, faces, points):
        # verts B*V*3
        # faces 1*F*3
        # points B*17*3
        # cam_trans B*3
        sil, p2d = self.render(vertices=verts, faces=faces.repeat(verts.shape[0], 1, 1), points=points)
        silhouette = sil[..., 3].unsqueeze(1)  # [N,1,256,256]
        point2d = p2d  # [N,17,2]
        return silhouette, point2d

    def get_color_results(self, verts, faces):
        # verts B*V*3
        # faces 1*F*3
        # points B*17*3
        # cam_trans B*3
        color_img = self.render.get_color_image(vertices=verts,faces=faces.repeat(verts.shape[0], 1, 1))  # [N,3,256,256]
        return color_img

    def forward(self, batch, batch_idx):
        if self.opts.TEXT and not self.opts.GT and self.opts.useSynData:
            input_image, mask_gt, _, _, _, _, _, _, _, label_tensor, frame_idx, kp2d = (*map(lambda x: validate_tensor_to_device(x, self.device), batch),)
        elif self.opts.TEXT and self.opts.GT and self.opts.useSynData:
            input_image, mask_gt, _, _, _, _, _, _, _, label_tensor, frame_idx, kp2d, shape_gt_tensor, pose_gt_tensor, trans_gt_tensor = (
            *map(lambda x: validate_tensor_to_device(x, self.device), batch),)
        else:
            raise NotImplementedError
        
        if self.opts.useSynData:
            ##### syn dataset #####
            input_image = collapseBF(input_image)
            mask_gt = collapseBF(mask_gt)
            kp2d = collapseBF(kp2d)
            if self.opts.GT:
                shape_gt_tensor = collapseBF(shape_gt_tensor)
                pose_gt_tensor = collapseBF(pose_gt_tensor)
                trans_gt_tensor = collapseBF(trans_gt_tensor).squeeze(1) #[batch, data_batch_size, 1,3] --> [N, 1, 3]
            else:
                shape_gt_tensor, pose_gt_tensor, trans_gt_tensor = torch.FloatTensor([float('nan')]), torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')])
        else:
            raise NotImplementedError
        
        if self.opts.ModelName in self.HMRKEYS:
            pred_rotmat, pred_betas, pred_cam_crop = self.model(input_image)
        elif self.opts.ModelName in self.DISENKEYS:
            pred_rotmat, pred_betas, pred_cam_crop, xf_shape, xf_pose, xf_cam = self.model(input_image,)

        # adapt the label to the shape of the keypoints and loss calculateion
        if self.opts.DatasetName == 'DessiePIPEWithRealImage':
            label_flatten_tensor = collapseBF(label_tensor) #[N,2] --> [2N]
            label_new_tensor = label_tensor[:,[0]]
        else:
            label_flatten_tensor = collapseBF(label_tensor.repeat(1,self.opts.data_batch_size)) if self.opts.getPairs else collapseBF(label_tensor) #[N,1] --> [N,2] --> [2N] 
            label_new_tensor = label_tensor[:,[0]] if not self.opts.getPairs else label_tensor
        
        # convert the camera parameters from the crop camera
        pred_trans_cam_crop = self.cam_bbox(pred_cam_crop, r=self.img_size[0], focal_length=self.FOCAL_LENGTH_BBOX)
        pred_vertices_crop, _, _ = self.get_SMAL_results(pred_betas, pred_rotmat, pred_trans_cam_crop)
        pred_kp3d_crop_selected = get_point(pred_vertices_crop, flag = 'COMBINAPOINTINDEX')
        pred_kp3d_crop = organize_kp(pred_kp3d_crop_selected, label_flatten_tensor)
        predseg_crop, predp2d_crop = self.get_render_results(verts=pred_vertices_crop, faces=self.faces,points=pred_kp3d_crop)

        if (label_flatten_tensor==5).any(): # remove the last point for animal3d dataset
            predp2d_crop[label_flatten_tensor==5,16,]= 0.
        denormalized_image = input_image * self.image_std.view(1, 3, 1, 1) + self.image_mean.view(1, 3, 1, 1)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                color_image = self.get_color_results(verts=pred_vertices_crop, faces=self.faces)
                color_image = torch.cat((denormalized_image, color_image), dim=3)
        else:
            color_image = None

        if self.opts.ModelName in self.DISENKEYS and self.opts.useSynData and self.opts.getPairs:
            xf_shape_batch = expandBF(xf_shape, self.opts.batch_size, self.opts.data_batch_size)
            xf_pose_batch = expandBF(xf_pose, self.opts.batch_size, self.opts.data_batch_size)
            xf_cam_batch = expandBF(xf_cam, self.opts.batch_size, self.opts.data_batch_size)
            pred_betas_batch = expandBF(pred_betas, self.opts.batch_size, self.opts.data_batch_size)
            pred_rotmat_batch = expandBF(pred_rotmat, self.opts.batch_size, self.opts.data_batch_size)
            pred_trans_batch = expandBF(pred_trans_cam_crop, self.opts.batch_size, self.opts.data_batch_size)
        else:
            xf_shape_batch,xf_pose_batch,xf_cam_batch = torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')])
            pred_betas_batch,pred_rotmat_batch,pred_trans_batch = torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')])
            xf_shape,xf_pose,xf_cam = torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')]),torch.FloatTensor([float('nan')]) #if self.opts.ModelName in self.HMRKEYS:
        return {'pred_betas': pred_betas, 'pred_rotmat': pred_rotmat, 'pred_trans_cam_crop': pred_trans_cam_crop,
                'pred_kp2d_crop': predp2d_crop, 'pred_mask_crop': predseg_crop, 'pred_color_img': color_image,
                'gt_kp2d': kp2d, 'gt_mask_crop': mask_gt, 'input_image': input_image, 'batch_idx': batch_idx,
                'labels': label_tensor, 'xf_shape_batch': xf_shape_batch, 'xf_pose_batch': xf_pose_batch,'xf_cam_batch': xf_cam_batch,
                'xf_shape': xf_shape, 'xf_pose': xf_pose, 'xf_cam': xf_cam,
                'denormalized_image':denormalized_image,
                'pred_kp3d_crop':pred_kp3d_crop,'pred_vertices_crop':pred_vertices_crop,
                'gt_shape':shape_gt_tensor, 'gt_pose': pose_gt_tensor, 'gt_trans': trans_gt_tensor,
                'pred_betas_batch':pred_betas_batch, 'pred_rotmat_batch':pred_rotmat_batch, 'pred_trans_batch':pred_trans_batch,
                'label_flatten':label_flatten_tensor, 'label_batch': label_new_tensor
         }

    def loss(self, data):
        '''Compute stochastic loss.'''
        obj = {}
        obj['shape'] = self.opts.W_shape_prior * (data['pred_betas'] ** 2).sum(dim=-1).mean()  # betas_loss(data['pred_betas']) betas_loss(data['pred_betas'])
        if (data['gt_kp2d'][:,:,2]>0.5).sum()>0:
            obj['kp'] = self.opts.W_kp_img * kp2d_loss(data['pred_kp2d_crop'], data['gt_kp2d'])
        else:
            obj['kp'] = torch.tensor(0.0, device=data['gt_kp2d'].device)
        obj['pose'] = self.opts.W_pose_img * self.prior(data['pred_rotmat']).mean()
        ###### Animal3D without mask loss #######
        if (data['label_flatten']!=5).sum() > 0:
            obj['mask'] = self.opts.W_mask_img * mask_loss(proj_masks=data['pred_mask_crop'][data['label_flatten']!=5], masks=data['gt_mask_crop'][data['label_flatten']!=5])
        else:
            obj['mask'] = torch.tensor(0.0, device=data['gt_mask_crop'].device)
        ###### All data have mask loss #######
        if self.opts.ModelName in self.DISENKEYS and self.opts.getPairs:  # self.opts.useIMAGEPAIR == 'class3':
            obj['L21_shape_label1'] = self.opts.W_l2_shape_1 * latentL2_loss(data['xf_shape_batch'], data['label_batch'] == 1)  ## label to 1: pose space: change pose ; shape/cam equal
            obj['L22_rootrot_label1'] = self.opts.W_l2_rootrot_1 *  latentL2_loss(data['pred_rotmat_batch'][:,:,[0],...], data['label_batch'] == 1)  ## label to 1: pose space: change pose ; shape/cam equal
            
            obj['L21_pose_label2'] = self.opts.W_l2_pose_2 * latentL2_loss(data['xf_pose_batch'], data['label_batch'] == 2)  ## ; label to 2: appearance space: change  ; pose/cam equal
            obj['L22_rootrot_label2'] = self.opts.W_l2_rootrot_2 *  latentL2_loss(data['pred_rotmat_batch'][:,:,[0],...], data['label_batch'] == 2) ## label to 2: appearance space: change appearance ; pose/cam equal
            
            obj['L21_shape_label3'] = self.opts.W_l2_shape_3 * latentL2_loss(data['xf_shape_batch'], data['label_batch'] == 3)  ## ; label to 3: cam space: change cam ; shape/pose equal
            obj['L21_pose_label3'] = self.opts.W_l2_pose_3 * latentL2_loss(data['xf_pose_batch'], data['label_batch'] == 3)  ## ; label to 3: cam space: change cam ; shape/pose equal
        if self.opts.GT and self.opts.pred_trans:
            mask = (data['label_flatten'] != 4) & (data['label_flatten'] != 5)
            if mask.sum() > 0:
                obj['gt_shape'] = self.opts.W_gt_shape * gt_loss(pred = data['pred_betas'][mask], gt = data['gt_shape'][mask])
                obj['gt_pose']  = self.opts.W_gt_pose * gt_loss(pred = data['pred_rotmat'][mask], gt = batch_rodrigues(data['gt_pose'][mask].view(-1,3)).view(-1,36,3,3)) 
                obj['gt_trans'] = self.opts.W_gt_trans * gt_loss(pred = data['pred_trans_cam_crop'][mask], gt = data['gt_trans'][mask])
            else:
                obj['gt_shape'] = torch.tensor(0.0, device=data['gt_shape'].device)
                obj['gt_pose'] = torch.tensor(0.0, device=data['gt_pose'].device)
                obj['gt_trans'] = torch.tensor(0.0, device=data['gt_trans'].device)
        return obj

    def on_train_start(self):
        self.initial_setup()
        # ... other operations ...

    def on_validation_start(self):
        self.initial_setup()
        # ... other operations ...

    def training_step(self, x_batch, batch_idx):
        # x_batch = self._get_features(batch)
        data = self.forward(x_batch, batch_idx)
        obj = self.loss(data)
        loss = reduce(lambda x, y: x + y, obj.values())
        # # Log individual losses
        for key, value in obj.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)  # Lightning logs batch-wise metrics during training per default
        self.log('train/loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Check if the current iteration is a multiple of 100
        if batch_idx % 10 == 0 and self.trainer.global_rank == 0:
            self.log_all(logger_prefix='train', data=data)
        return {'loss': loss, 'log': obj}

    def validation_step(self, x_batch, batch_idx):
        # x_batch = self._get_features(batch)
        data = self.forward(x_batch, batch_idx)
        obj = self.loss(data)
        loss = reduce(lambda x, y: x + y, obj.values())
        # Log individual losses
        for key, value in obj.items():
            self.log(f'val/{key}', value, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                     sync_dist=True)  # Lightning automatically averages metrics over batches for validation
        self.log('val/loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # on_step: record each step, on_epoch: record each epoch
        # Check if the current iteration is a multiple of 100
        if batch_idx % 10 == 0 and self.trainer.global_rank == 0:
            self.log_all(logger_prefix='val', data=data)
        return {'loss': loss, 'log': obj}

    def test_step(self, x_batch, batch_idx):
        # x_batch = self._get_features(batch)
        data = self.forward(x_batch, batch_idx)
        obj = self.loss(data)
        loss = reduce(lambda x, y: x + y, obj.values())
        # Log individual losses
        for key, value in obj.items():
            self.log(f'test/{key}', value, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)  # Lightning automatically averages metrics over batches for testing
        self.log('test/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Check if the current iteration is a multiple of 100
        if batch_idx % 10 == 0 and self.trainer.global_rank == 0:
            self.log_all(logger_prefix='test', data=data)
        return {'loss': loss, 'data': data}

    def test_epoch_end(self, outputs):
        dic = {}
        for k in outputs[0]['data'].keys():
            dic[k] = []

        for output in outputs:
            for k in output['data'].keys():
                if k not in dic.keys():
                    dic[k] = []
                data = list(output['data'][k].cpu().detach().numpy())  # predicted values
                dic[k].extend(data)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        dic['loss'] = avg_loss.cpu().detach().numpy()

        # checkpoint_path = self.trainer.ckpt_connector.checkpoint_path
        checkpoint_path = self.checkpoint_name
        print(f"Currently loaded checkpoint: {checkpoint_path}")
        torch.save(dic, os.path.join(os.path.dirname(checkpoint_path), #self.logger.log_dir,
                                     f"{os.path.basename(checkpoint_path).split('.')[0]}_predictions.pt"))

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def _get_features(batch):
        '''Get only batch features and discard the rest.'''
        if isinstance(batch, (tuple, list)):
            x_batch = batch[0]
        elif isinstance(batch, dict):
            x_batch = batch['features']
        elif isinstance(batch, torch.Tensor):
            x_batch = batch
        else:
            raise TypeError('Invalid batch type encountered: {}'.format(type(batch)))
        return x_batch

    def log_image(self, logger_prefix, name, image):
        self.logger.experiment.add_image(logger_prefix + '/' + name, image_grid(image.detach().cpu().clamp(0, 1)),global_step=self.global_step)  # self.current_epoch) #

    def log_video(self, logger_prefix, name, frames):
        self.logger.add_video(logger_prefix + '/animation' + name, frames.detach().cpu().unsqueeze(0).clamp(0, 1),self.current_epoch, fps=2)

    def log_all(self, logger_prefix, data):
        overlap = (data['gt_mask_crop'][:4, ...] + data['pred_mask_crop'][:4, ...]) / 2.  # [4,1,256,256]
        self.log_image(logger_prefix, 'overlap_mask', overlap.repeat(1, 3, 1, 1))
        if data['pred_color_img'] is not None:
            pred_img = data['pred_color_img'][:4, ...]
            self.log_image(logger_prefix, 'pred_img', pred_img)