'''
Evalutation code -- Code adapted from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/benchmark/pascal/kp_project.py
'''

from __future__ import absolute_import, division, print_function
import numpy as np
from absl import app, flags
import torch
import torchvision
from torch.utils.data import DataLoader

import sys,os, random, math
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

from src.train2 import parse_args
from src.test import get_model
from src.staths.objects import ImageDatasetEval
from einops import rearrange
import logging
from src.render.renderer_pyrender import Renderer
from PIL import Image

def none_to_nan(x):
    if x is None:
        return torch.FloatTensor([float('nan')])
    elif isinstance(x, int):
        return torch.FloatTensor([x])
    else:
        return x

class Evaluator():

    def __init__(self, opts, device):
        self.opts = opts
        self.device = device
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.Tensor
        self.expand = opts.expand
        self.mask_background = opts.mask_background

    def init_dataset(self):
        opts = self.opts
        # initialize the dataset
        dset = ImageDatasetEval(opts)
        self.dataloader = DataLoader(dset, batch_size = opts.batch_size, num_workers = opts.n_data_workers, pin_memory = True)
        self.resnet_transform = torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )

    def define_model(self, args):
        self.Model, trainer = get_model(args, args.ckpt_file, self.device)
        self.Model.initial_setup()
        ####setup render####
        self.faces_cpu = self.Model.smal_model.faces.unsqueeze(0).cpu().data.numpy()[0]
        self.render = Renderer(focal_length=5000, img_w=256, img_h=256, faces=self.faces_cpu, same_mesh_color=False)   

    def set_input(self, batch):
        # get joints
        if 'kp' in batch:
            kp = batch['kp'].type(self.Tensor).to(self.device)
        # get masks
        if self.opts.mask_anno and 'mask' in batch:
            mask = batch['mask'].type(self.Tensor)
            mask = (mask > 0.5).float()
            self.mask = mask.to(self.device)
        # get input images
        input_imgs = batch['img'].type(self.Tensor)
        input_imgs_denormalized = batch['img'].type(self.Tensor).clone()
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.imgs = input_imgs.to(self.device)
        # get pseudo-gt cameras from sfm
        if self.opts.sfm_anno and 'sfm_pose' in batch:
            self.quat_gt = batch['sfm_pose'][2].type(self.Tensor)
            self.valid_cam = batch['valid_cam'].numpy()
        else:
            self.quat_gt = none_to_nan(None)
            self.valid_cam = none_to_nan(None)
        if self.expand:
            self.imgs = self.imgs.unsqueeze(0)
            self.mask = self.mask.unsqueeze(0)
            kp = kp.unsqueeze(0)
            input_imgs_denormalized = input_imgs_denormalized.unsqueeze(0)
        self.batch = (
            self.imgs, self.mask, input_imgs_denormalized, torch.FloatTensor([[0]]), torch.FloatTensor([[0]]), torch.FloatTensor([[0]]),
            torch.FloatTensor([[0]]), none_to_nan(None), none_to_nan(None), none_to_nan(None),  none_to_nan(None),
            kp)

    def get_predict_kp(self, pred_vertices):
        from src.utils.model_utils import get_point
        pred_3D_kp_temp = get_point(pred_vertices, 'ANIMAL3D')
        pred_2d_kp_crop = self.Model.render.camera.transform_points_screen(pred_3D_kp_temp, image_size=torch.ones(1, 2).to(pred_vertices.device) * self.opts.img_size)[:, :, :2]
        pred_2d_kp_crop_norm = self.normalize_joints(pred_2d_kp_crop, self.opts.img_size)
        return pred_2d_kp_crop, pred_2d_kp_crop_norm

    def normalize_joints(self, joints, img_size):
        new_joints = 2 * (joints / img_size) - 1
        return new_joints

    def predict(self, save_path = None, i = None):
        with torch.no_grad():
            pred_data = self.Model(batch = self.batch, batch_idx = 0)
        self.codes_pred = pred_data
        self.codes_pred['kp_project_selected'],self.codes_pred['kp_project_selected_norm'] = self.get_predict_kp(pred_data['pred_vertices_crop'])
        '''the mask'''
        self.codes_pred['mask_predicted_for_evaluation'] = self.render(pred_data['pred_vertices_crop'][0].cpu().data.numpy(), image = None, obtainSil = True) 
        '''visualize'''
        if save_path is not None:
            img = pred_data['pred_color_img'].cpu().numpy().transpose(0, 2, 3, 1)
            img = torchvision.transforms.ToPILImage()((img[0]*255).astype(np.uint8))
            img.save(os.path.join(save_path, f'{self.opts.dataset}_{str(i).zfill(7)}.jpg'))
            # img.show()

    def evaluate(self):
        ## Keypoint reprojection error
        padding_frac = self.opts.padding_frac
        # The [-1,1] coordinate frame in which keypoints corresponds to:
        #    (1+2*padding_frac)*max_bbox_dim in image coords
        # pt_norm = 2* (pt_img - trans)/((1+2*pf)*max_bbox_dim)
        # err_pt = 2*err_img/((1+2*pf)*max_bbox_dim)
        # err_pck_norm = err_img/max_bbox_dim = err_pt*(1+2*pf)/2
        # so the keypoint error in the canonical frame should be multiplied by:
        err_scaling = (1 + 2 * padding_frac) / 2.0
        if self.opts.expand:
            kps_gt = rearrange(self.batch[11], 'b f ... -> (b f) ...').cpu().numpy()
        else:
            kps_gt = self.batch[11].cpu().numpy()
        kps_vis  = kps_gt[:, :, 2]
        kps_gt   = kps_gt[:, :, 0:2]
        #####################
        kps_pred = self.codes_pred['kp_project_selected_norm'].type_as(self.batch[11]).cpu().numpy()
        kps_err  = kps_pred - kps_gt
        kps_err  = np.sqrt(np.sum(kps_err * kps_err, axis=2)) * err_scaling

        ## mIoU metric
        iou = None
        if self.opts.mask_anno and self.mask is not None:
            mask = self.mask
            bs = mask.size(0)
            mask_gt = mask.view(bs, -1).cpu().numpy()
            mask_pred = self.codes_pred['mask_predicted_for_evaluation'] 
            mask_pred = mask_pred.astype(np.float32).reshape(bs,-1)
            intersection = mask_gt * mask_pred
            union = mask_gt + mask_pred - intersection
            iou = intersection.sum(1) / union.sum(1)

        # ## camera rotation error
        quat_error, azel_gt, azel_pred = None, None, None
        return kps_err, kps_vis, iou, quat_error, azel_gt, azel_pred


    def compute_metrics(self, stats, args = None):
        logger = logging.getLogger()  # Gets the root logger
        logger.setLevel(logging.INFO)
        # If the logger has handlers, remove them (this prevents duplicating logs)
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])
        #######################
        if args is not None:
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt_file))), f"Evaluate_2d_{self.opts.dataset}_{args.ckpt_file.split('/')[8]}_eval.log")
        
            file_handler = logging.FileHandler(save_path, mode='w')
            console_handler = logging.StreamHandler()    
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Log arguments
            args_dict = vars(args)
            args_str = ', '.join(f'{key}={value}' for key, value in args_dict.items())
            logging.info(f'Running with arguments: {args_str}')
            logging.info('%s',args.ModelName)
            logging.info('%s',args.ckpt_file)
        
        # AUC
        n_vis_p          = np.sum( stats['kps_vis'] )
        n_correct_p_pt06 = np.sum( (stats['kps_err'] < 0.06) * stats['kps_vis'])
        n_correct_p_pt07 = np.sum( (stats['kps_err'] < 0.07) * stats['kps_vis'])
        n_correct_p_pt08 = np.sum( (stats['kps_err'] < 0.08) * stats['kps_vis'])
        n_correct_p_pt09 = np.sum( (stats['kps_err'] < 0.09) * stats['kps_vis'])
        n_correct_p_pt10 = np.sum( (stats['kps_err'] < 0.10) * stats['kps_vis'])
        pck06 = 100 * (n_correct_p_pt06 / n_vis_p)
        pck07 = 100 * (n_correct_p_pt07 / n_vis_p)
        pck08 = 100 * (n_correct_p_pt08 / n_vis_p)
        pck09 = 100 * (n_correct_p_pt09 / n_vis_p)
        pck10 = 100 * (n_correct_p_pt10 / n_vis_p)
        auc   = (pck06+pck07+pck08+pck09+pck10) / 5
        if args is not None:
            logging.info('%s %.1f', '=> pck06:', pck06)
            logging.info('%s %.1f', '=> pck07:', pck07)
            logging.info('%s %.1f', '=> pck08:', pck08)
            logging.info('%s %.1f', '=> pck09:', pck09)
            logging.info('%s %.1f', '=> pck10:', pck10) 
            logging.info('%s %.1f', '=> AUC:', auc)

        # Camera error
        if self.opts.sfm_anno:
            mean_cam_err = stats['quat_error'].sum() / sum(stats['quat_error']>0)
            print('=> cam_err: {:.1f}'.format(mean_cam_err))

        # mIoU
        if self.opts.mask_anno:
            mean_iou = 100 * stats['ious'].mean()
            if args is not None:
                logging.info('%s %.1f', '=> mIOU ', mean_iou)


    def test(self, save_path = None, args = None):
        print(f'Evaluating on {self.opts.dataset} ...')
        bench_stats = { 'kps_err': [], 'kps_vis': [], 'ious' : [], 'quat_error': [], 'azel_gt': [], 'azel_pred': [] }
        for i,batch in enumerate(self.dataloader):
            self.set_input(batch)
            self.predict(save_path=save_path, i=i)
            kps_err, kps_vis, iou, quat_error, azel_gt, azel_pred = self.evaluate()
            bench_stats['kps_err'].append(kps_err)
            bench_stats['kps_vis'].append(kps_vis)
            if self.opts.mask_anno:
                bench_stats['ious'].append(iou)

        bench_stats['kps_err'] = np.concatenate(bench_stats['kps_err'])
        bench_stats['kps_vis'] = np.concatenate(bench_stats['kps_vis'])
        if self.opts.mask_anno:
            bench_stats['ious'] = np.concatenate(bench_stats['ious'])
        if self.opts.sfm_anno:
            bench_stats['quat_error'] = np.concatenate(bench_stats['quat_error'])
        self.compute_metrics(bench_stats, args= args)
        
def main():
    class Options:
        def __init__(self, dataset, category, kp_anno, mask_anno, sfm_anno,batch_size, n_data_workers, img_size,
                     jitter_frac, padding_frac, tight_crop, split, data_dir, expand, mask_background):
            self.dataset = dataset
            self.category = category
            self.kp_anno = kp_anno
            self.mask_anno = mask_anno
            self.sfm_anno = sfm_anno
            self.batch_size = batch_size  
            self.n_data_workers = n_data_workers
            self.img_size = img_size
            self.padding_frac = padding_frac    # Don't pad/jitter while evaluating
            self.jitter_frac  = jitter_frac    # Don't pad/jitter while evaluating
            self.tight_crop = tight_crop
            self.split = split
            self.data_dir = data_dir
            self.expand = expand
            self.mask_background = mask_background

    args = parse_args()
    print(args)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    # Run evaluation
    SAVE_IMG = False
    if SAVE_IMG == True:
        save_img_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_file)), f"Evaluate_2d")
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        print(save_img_path)
        save_img_path_flag = save_img_path
    else:
        save_img_path_flag = None

    img_size = args.imgsize
    ###########pascal##################################################################
    opts = Options(dataset="pascal", category="horse", kp_anno=True, mask_anno=True, sfm_anno=False,
                   batch_size=1, n_data_workers=4, img_size=img_size, jitter_frac = 0, padding_frac = 0,tight_crop = False,
                   split='val', data_dir = args.data_dir, expand=True, mask_background = False)    
    assert (opts.category in ['horse', 'cow', 'sheep'] and opts.dataset in ['pascal', 'animal_pose']) or \
                opts.category in ['giraffe', 'bear'] and opts.dataset=='coco', 'Error in category/dataset arguments'
    tester = Evaluator(opts, device)
    tester.init_dataset()
    tester.define_model(args)
    tester.test(save_path = save_img_path_flag, args =args)

    ###########animal_pose##################################################################
    opts = Options(dataset="animal_pose", category="horse", kp_anno=True, mask_anno=True, sfm_anno=False,
                   batch_size=1, n_data_workers=4, img_size=img_size, jitter_frac = 0, padding_frac = 0,tight_crop = False,
                   split='val', data_dir = args.data_dir, expand=True,mask_background = False)    
    assert (opts.category in ['horse', 'cow', 'sheep'] and opts.dataset in ['pascal', 'animal_pose']) or \
                opts.category in ['giraffe', 'bear'] and opts.dataset=='coco', 'Error in category/dataset arguments'
    tester = Evaluator(opts, device)
    tester.init_dataset()
    tester.define_model(args)
    tester.test(save_path = save_img_path_flag,args = args)

    #########################################################################################

if __name__ == '__main__':
    # app.run(main)
    main()