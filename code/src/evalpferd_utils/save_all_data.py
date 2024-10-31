import sys, os, torch, cv2, copy,re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use('TkAgg')
from src.utils.misc import validate_tensor_to_device, validate_tensor, collapseBF
from src.test import get_model
from src.train2 import parse_args
from src.render.renderer_pyrender import Renderer
from PIL import Image
from src.evalpferd_utils.pferd import PFERD
import os.path as osp
import numpy as np
from src.SMAL.smal_torch.smal_torch import SMAL
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parse_args()
    #####################set data##################################################
    
    videodir = {'20201128_ID_2_0010_Miqus_65_20715': [4,15],
                '20201128_ID_2_0008_Miqus_61_23417': [15,20],
                '20201128_ID_1_0001_Miqus_50_23416': [0,8],
                '20201129_ID_4_0005_Miqus_64_23414': [0,5],
                '20201129_ID_4_0008_Miqus_64_23414': [32,36],
    }
    
    videoskeys = sorted(videodir.keys())
    for i in range(len(videoskeys)):
        videonames = videoskeys[i]
        start_time = videodir[videonames][0]
        end_time = videodir[videonames][1]
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("========== Preparing model... ========== ")
        Model, trainer = get_model(args, args.ckpt_file, device)
        Model.initial_setup()
        
        # Check the mode right after loading the model
        if Model.training:
            print("Model is in training mode.")
        else:
            print("Model is in evaluation mode.")
        
        print("========== Preparing data... ========== ")
        traindataset = PFERD(data_dir= args.data_dir,
                                    videonames = videonames, start_time = start_time, end_time = end_time,)
        train_dl = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=1)
            
        SAVE = True
        if SAVE:
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt_file))))), 'PFERD', 'Dessie', f'DATA_{videonames}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            gt = {'betas':[], 'poses': [], 'trans': [], 'vertices':[]} 
            pred = {'betas':[], 'poses': [], 'trans': [], 'vertices':[]}
            
            gt['R']  = traindataset.cam['R']
            gt['T']  = traindataset.cam['T']
            
        print("========== Preparing render... ========== ")
        ####setup render####
        faces_cpu = Model.smal_model.faces.unsqueeze(0).cpu().data.numpy()[0]
        render = Renderer(focal_length=5000, img_w=args.imgsize, img_h=args.imgsize, faces=faces_cpu, same_mesh_color=False)    
        
        print("===========  initial hSMAL model ===========")
        smal_model = SMAL(os.path.join(args.model_dir, 'my_smpl_0000_horse_new_skeleton_horse.pkl'),device=device, use_smal_betas = True)
        
        print("=========== start testing ===========")
        with torch.no_grad():
            for i, data in enumerate(train_dl):
                input_image_tensor = data[0] #[1,1,3,256,256] --> [1,3,256,256]
                denormalized_image_tensor = data[2] #[1,1,3,256,256] --> [1,3,256,256]
                input_image_tensor = collapseBF(input_image_tensor).to(device)
                denormalized_image_tensor = collapseBF(denormalized_image_tensor)
                gt_betas = collapseBF(data[12]).to(device) #[1,1,10] --> [1,10]
                gt_poses = collapseBF(data[13]).to(device) #[1,1,108] --> [1,108]
                gt_trans = collapseBF(data[14]).to(device) #[1,1,3] --> [1,3]
                
                ### predict
                pred_data = Model.predict(input_image_tensor)

                ### predict SMAL and joint
                pred_vertices,  pred_joint, _ = Model.get_SMAL_results(betas = pred_data['pred_betas'], 
                                                                        poses = pred_data['pred_rotmat'], 
                                                                        trans = pred_data['pred_trans_cam_crop'])
                ### gt SMAL and gt joint
                gt_vertices, gt_joint, _ = smal_model(beta = gt_betas, 
                                                    theta = gt_poses, 
                                                    trans = gt_trans)
                # need to rotate the vertices to be in the camera coordinate system
                rotation = torch.from_numpy(traindataset.cam['R'].copy()).float().to(device) # Convert to torch tensor if in NumPy
                translation = torch.from_numpy(traindataset.cam['T'].copy()).float().to(device) # Convert to torch tensor if in NumPy
                # Rotate vertices: Shape of rotation is [3, 3], vertices is [1, 1479, 3]
                gt_rotated_vertices = torch.matmul(gt_vertices, rotation.T)+translation  # Shape: [1, 1479, 3] 
                gt_rotated_joint =  torch.matmul(gt_joint, rotation.T)+translation  # Shape: [1, 1479, 3] 

                ### save video
                if SAVE:
                    gt['betas'].append(gt_betas.cpu().numpy())
                    gt['poses'].append(gt_poses.cpu().numpy())
                    gt['trans'].append(gt_trans.cpu().numpy())
                    gt['vertices'].append(gt_vertices.cpu().numpy())
                    
                    pred['betas'].append(pred_data['pred_betas'].cpu().numpy())
                    pred['poses'].append(pred_data['pred_rotmat'].cpu().numpy())
                    pred['trans'].append(pred_data['pred_trans_cam_crop'].cpu().numpy())
                    pred['vertices'].append(pred_data['pred_vertices_crop'].cpu().numpy())
            version = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt_file)))
            np.savez(f'{save_path}/{args.ModelName}_{args.name}_{version}_gt.npz', **gt)
            np.savez(f'{save_path}/{args.ModelName}_{args.name}_{version}_pred.npz', **pred)