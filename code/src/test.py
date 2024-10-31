'''
Test
'''

import os.path as osp
import glob
import cv2
import numpy as np
from scipy.io import loadmat
import torch
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from src.utils.misc import collapseBF
from src.train2 import parse_args
from src.render.renderer_pyrender import Renderer
from PIL import Image
from src.trainer.vanillaupdatetrainer import VanillaUpdateTrainer

def get_model(args,ckpt_file, device):
    # initialize model
    if args.TEXT:
        vanilla = VanillaUpdateTrainer(opts=args)
    else:
        raise NotImplementedError
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    print("Model saved at epoch:", checkpoint['epoch'])
    Model = vanilla.load_from_checkpoint(ckpt_file, opts=args)
    Model = Model.eval()
    Model = Model.to(device)
    return Model, vanilla

def set_default_args(args):
    ########################same for all model#######################################
    args.DatasetName='DessiePIPE'
    args.name='TOTALRANDOM'
    args.useSynData= True 
    args.TEXT = True 
    args.DINO_model_name='dino_vits8' 
    args.imgsize=256
    args.GT = True 
    args.pred_trans = True 
    args.W_shape_prior=0.0 
    args.W_kp_img=0.0 
    args.W_pose_img=0.0 
    args.W_mask_img=0.0
    args.W_cos_shape=0. 
    args.W_cos_pose=0. 
    args.W_text_shape=0. 
    args.W_text_pose=0. 
    args.W_text_cam=0. 
    args.W_cosine_text_shape=0.0 
    args.W_cosine_text_pose=0.0 
    args.W_cosine_text_cam=0.0 
    args.W_gt_shape=0. 
    args.W_gt_pose=0. 
    args.W_gt_trans=0. 
    args.W_l2_shape_1=0.0 
    args.W_l2_pose_2=0.0 
    args.W_l2_shape_3=0.0 
    args.W_l2_pose_3=0.0
    args.W_l2_rootrot_1=0.0 
    args.W_l2_rootrot_2=0.0 
    args.batch_size = 1
    args.data_batch_size=1
    args.getPairs = True 
    return args

class ImageDataset(Dataset):
    def __init__(self, path='/home/x_cili/x_cili_lic/DESSIE/data/demo'):
        self.path = path
        self.img_files = sorted(glob.glob(osp.join(self.path, '*.jpg')))
        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.norm_transform = transforms.Normalize(img_mean, img_std)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = self.img_files[idx]
        img = cv2.imread(img) / 255.
        img = img[:, :, ::-1]
        w,h = img.shape[1], img.shape[0]    
        if w != 256 or h != 256:
            img = cv2.resize(img, (256, 256))
        denormalized_tensor = torch.from_numpy(img.copy()).permute(2,0,1)[None,:,:,:].float()
        img_tensor = self.norm_transform(denormalized_tensor)

        out  = (img_tensor, torch.FloatTensor([[0]]), denormalized_tensor,torch.FloatTensor([[0]]), torch.FloatTensor([[0]]), torch.FloatTensor([[0]]),
            torch.FloatTensor([[0]]), torch.FloatTensor([[0]]),torch.FloatTensor([[0]]),torch.FloatTensor([[0]]),torch.FloatTensor([[0]]),
            torch.FloatTensor([[0]]))
        return out
    
if __name__ == '__main__':
    args = parse_args()
    args = set_default_args(args=args)
    
    ###### choose the model####
    args.ModelName = 'DESSIE'
    EFOLDER = 'COMBINAREAL'; version = 8 
    args.ckpt_file= f'/home/x_cili/x_cili_lic/DESSIE/results/model/{EFOLDER}/version_{version}/checkpoints/best.ckpt'
    
    ##### choose the model ####
    # args.ModelName = 'DINOHMR'
    # EFOLDER = 'COMBINAREAL'; version = 9 
    # args.ckpt_file= f'/home/x_cili/x_cili_lic/DESSIE/results/model/{EFOLDER}/version_{version}/checkpoints/best.ckpt'
    
    
    #### choose data####
    args.data_dir =  '/home/x_cili/x_cili_lic/DESSIE/data/demo'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("========== Preparing model... ========== ")
    Model, trainer = get_model(args, args.ckpt_file, device)
    Model.initial_setup()
    save_path = f'/home/x_cili/x_cili_lic/DESSIE/results/demo'
    if not osp.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
        
    # get the dataset
    dataset = ImageDataset(args.data_dir)
    print(len(dataset))
    # get the data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    
    ####setup render####
    faces_cpu = Model.smal_model.faces.unsqueeze(0).cpu().data.numpy()[0]
    render = Renderer(focal_length=5000, img_w=256, img_h=256, faces=faces_cpu, same_mesh_color=False)    
   
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_image_tensor, _, denormalized_image_tensor, _, _, _, _, _, _, _, _, _ = batch
            input_image_tensor = collapseBF(input_image_tensor).to(device)
            denormalized_image_tensor = collapseBF(denormalized_image_tensor)
            if args.ModelName == 'DINOHMR':
                data = Model.latentpredict(input_image_tensor)
                xf = data['xf']
                pred_data = Model.easypredict(xf_shape=None, xf_pose=None, xf_cam=None, xf=xf, cameraindexclass_tensor = None, pose_encoded_text = None)
            else:        
                data = Model.latentpredict(input_image_tensor)
                xf_shape = data['xf_shape']
                xf_pose = data['xf_pose']
                xf_cam = data['xf_cam']
                pred_data = Model.easypredict(xf_shape=xf_shape, xf_pose=xf_pose, xf_cam=xf_cam,  xf=None, cameraindexclass_tensor = None, pose_encoded_text = None)
            
            img = pred_data['pred_color_img'].cpu()
            image = torch.cat([denormalized_image_tensor, img], dim=3)
            img = transforms.ToPILImage()((image[0]))
            img.save(os.path.join(save_path, f'{EFOLDER}_version_{version}_{str(i).zfill(7)}.jpg'))
            print("saved image: ", f'{str(i).zfill(7)}.jpg')

            original_image = denormalized_image_tensor[0].cpu().data.numpy().transpose(1,2,0)*255
            overlap_image = render(pred_data['pred_vertices_crop'][0].cpu().data.numpy(), image = original_image, obtainSil = False)
            color_image_Image = Image.fromarray((overlap_image).astype(np.uint8)).convert('RGB')
            color_image_Image.save(os.path.join(save_path,f'{EFOLDER}_version_{version}_overlap_{str(i).zfill(7)}.jpg'))