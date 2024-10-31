import sys, os, re 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.evalpferd_utils.data_preprocess_utils import get_bbox_from_kp2d, process_image
import matplotlib
matplotlib.use('TkAgg')
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os, torch
import numpy as np
import random
import os.path as osp
import glob
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw    
class PFERD(Dataset):
    def __init__(self, data_dir= '/home/x_cili/x_cili_lic/DESSIE/data/pferd',
                                videonames = '20201128_ID_2_0010_Miqus_65_20715', start_time= 0, end_time = 5):
        self.videonames = videonames
        self.ID = videonames.split('_')[2]
        self.camera = videonames.split('_')[-1]
        self.videoseq = videonames[:18]
        #self.image_path = osp.join(data_dir, f'ID_{self.ID}', 'IMAGE_DATA', self.videoseq, self.videonames)
        self.kp_path = osp.join(data_dir, f'ID_{self.ID}', 'KP2D_DATA', self.videoseq, f'{self.videonames}_2Dkp.npz')
        self.video_path = osp.join(data_dir, f'ID_{self.ID}', 'VIDEO_DATA', self.videoseq, f'{self.videonames}.avi')
        self.model_path = osp.join(data_dir, f'ID_{self.ID}', 'MODEL_DATA', f'{self.videoseq}_hsmal.npz')
        self.cam_path = osp.join(data_dir, f'ID_{self.ID}', 'CAM_DATA', f'Camera_Miqus_Video_{self.camera}.npz')
        ## load data
        self.ann = np.load(self.kp_path, allow_pickle=True) #'kp2d', 'kp3d', 'videoFps'
        self.model_ann = np.load(self.model_path, allow_pickle=True) #'betas''poses''trans',
        self.cam = np.load(self.cam_path, allow_pickle=True) #'R': rotation, 'T': translation, 'K': instrinsic parameters, 'D': Distortion parameters
    
        #self.image_list = sorted(glob.glob(osp.join(self.image_path,f'*.png')))
        self.videoFps = self.ann['videoFps'].item()
        self.kplabels = [ re.sub( r'ID\d_', '', i) for i in  self.ann['labels']]
        self.rigidmarker = [i for i, value in enumerate(self.kplabels) if (match := re.search(r'_(\d+)$', value)) and int(match.group(1)) <= 31]
        self.restmarker = [i for i, value in enumerate(self.kplabels) if not ((match := re.search(r'_(\d+)$', value)) and int(match.group(1)) <= 31)]
        self.betas_gt = self.model_ann['betas'].copy()
        interval = int(240/self.videoFps)
        poses_gt = self.model_ann['poses'][::interval,:].copy()
        trans_gt = self.model_ann['trans'][::interval,:].copy()
        assert poses_gt.shape[0] == self.ann['kp2d'].shape[0] == trans_gt.shape[0]
        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.norm_transform = transforms.Normalize(img_mean, img_std)
        self.frames, self.start_frame_index, self.end_frame_index = self._read_video_frames(self.video_path, start_time = start_time, end_time = end_time)
        ####### get data from start_time to end_time
        self.poses_gt = poses_gt[self.start_frame_index:self.end_frame_index]
        self.trans_gt = trans_gt[self.start_frame_index:self.end_frame_index]
        self.kp2d = self.ann['kp2d'][self.start_frame_index:self.end_frame_index]
        assert len(self.frames) == self.poses_gt.shape[0] == self.trans_gt.shape[0] == self.kp2d.shape[0]
        assert len(self.model_ann['missing_frame']) == 0, "missing frame in the video"

        
    def preprocess_image(self, img_BGR, bboxes, kp2d):
        center_ = [(bboxes[2] + bboxes[0]) / 2, (bboxes[3] + bboxes[1]) / 2] 
        scale_ = [1.2 * (bboxes[2] - bboxes[0]), 1.2 * (bboxes[3] - bboxes[1])]
        new_img_RGB, new_kp2d, _ ,_,_ = process_image(img_BGR, center_, 1.0 * np.max(scale_), kp2d[:,:2], seg =None)
        new_kp2d = np.hstack((new_kp2d,kp2d[:,2].reshape(-1,1)))
        wrongindex = np.unique(np.where(new_kp2d <= 0.)[0])
        new_kp2d[wrongindex] = 0.
        return new_img_RGB, new_kp2d


    def _read_video_frames(self, video_path, start_time, end_time):
        """
        Reads frames from the given .avi video file within the specified time range.

        Args:
            video_path (str): Path to the .avi video file.
            start_time (float): Start time in seconds to begin reading frames.
            end_time (float): End time in seconds to stop reading frames.

        Returns:
            frames (list): List of frames (each frame is a numpy array).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames

        # Get the video's frame rate (frames per second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == self.videoFps, f"fps is not equal to the videoFps in the annotation file" 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the start and end frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Make sure end_frame does not exceed the total number of frames in the video
        end_frame = min(end_frame, total_frames)

        # Set the video to the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read frames within the specified range
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break  # Break if no more frames can be read

            # Convert BGR (OpenCV default) to RGB (PIL/Image default)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()  # Release the video capture object
        return frames, start_frame, end_frame

    def __getitem__(self, item):
        frame_index = item
        img_BGR = self.frames[item]
        W,H = img_BGR.shape[1], img_BGR.shape[0]
        
        #################### get bbox ##########################
        kp2d_org = self.kp2d[frame_index].copy().astype(np.float32)
        # Initialize the new kp2d array with zeros, with shape (171, 3)
        kp2d = np.zeros((kp2d_org.shape[0], kp2d_org.shape[1] + 1))
        # Copy the original data to the new array, replacing NaNs with 0
        kp2d[:, :2] = np.nan_to_num(kp2d_org, nan=0)
        # Determine visibility: 1 if not NaN, 0 if NaN
        kp2d[:, 2] = ~np.isnan(kp2d_org).any(axis=1)  # Check NaN across both coordinates for each keypoint
        bboxes = get_bbox_from_kp2d(kp2d[:,:2]).astype(np.float32)
        
        #################### process images ######################
        new_img_RGB, new_kp2d = self.preprocess_image(img_BGR, bboxes, kp2d)
        new_img_RGB_01 = new_img_RGB.copy()/255.
        #################### get gt ##############################
        betas_gt = self.betas_gt
        poses_gt = self.poses_gt[frame_index] 
        trans_gt = self.trans_gt[frame_index]
        
        #################### prepare output ######################
        denormalized_tensor = torch.from_numpy(new_img_RGB_01.copy())[None,:,:,:].float()
        img_tensor = self.norm_transform(denormalized_tensor)
        keypoints = torch.from_numpy(new_kp2d.copy()).float().unsqueeze(0)
        bboxes_tensor = torch.from_numpy(bboxes.copy()).float().unsqueeze(0)
        
        betas_gt_tensor = torch.from_numpy(betas_gt.copy()).float().unsqueeze(0)
        pose_gt_tensor = torch.from_numpy(poses_gt.copy()).float().unsqueeze(0)
        trans_gt_tensor = torch.from_numpy(trans_gt.copy()).float().unsqueeze(0) 

        out  = (
            img_tensor, torch.FloatTensor([[0]]), denormalized_tensor,torch.FloatTensor([[0]]), torch.FloatTensor([[0]]), torch.FloatTensor([[0]]), #0-5
            torch.FloatTensor([[0]]), torch.FloatTensor([[0]]),torch.FloatTensor([[0]]), #6,7,8
            torch.FloatTensor([[0]]), bboxes_tensor,           keypoints, #9,10,11
            betas_gt_tensor, pose_gt_tensor, trans_gt_tensor)        #12,13,14
        
        return out

    def __len__(self):
        return len(self.frames)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print("========== Preparing data... ========== ")
    videodir = {'20201128_ID_2_0010_Miqus_65_20715': [4,15],
                '20201128_ID_2_0008_Miqus_61_23417': [15,20],
                '20201128_ID_1_0001_Miqus_50_23416': [0,8],
                '20201129_ID_4_0005_Miqus_64_23414': [0,5],
                '20201129_ID_4_0008_Miqus_64_23414': [32,36],
    }
    videonames = '20201128_ID_2_0010_Miqus_65_20715'
    start_time = videodir[videonames][0]
    end_time = videodir[videonames][1]
    traindataset = PFERD(data_dir=  '/home/x_cili/x_cili_lic/DESSIE/data/pferd', 
                                videonames = videonames, start_time= start_time, end_time = end_time)
    train_dl = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=1)
    
    # plt.figure()
    # plt.ion()
    
    # to create images for magicpony
    SAVE = True
    if SAVE:
        save_path = os.path.join('/home/x_cili/x_cili_lic/DESSIE/data/PFERD_Images/', videonames)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    for i, data in enumerate(train_dl):
        img_tensor = data[0][0] #[1,1,3,256,256] --> [1,3,256,256]
        denorm_tensor = data[2][0] #[1,1,3,256,256] --> [1,3,256,256]
        keypoints = data[11][0] #[1,1,17,3] --> [1,17,3]
        
        betas = data[12][0] #[1,1,10] --> [1,10]
        poses = data[13][0] #[1,1,108] --> [1,108]
        trans = data[14][0] #[1,1,3] --> [1,3]
        #import pdb; pdb.set_trace()        
        new_img_numpy = denorm_tensor[0,...].permute(1, 2, 0).cpu().data.numpy()
        new_kp2d_numpy = keypoints.cpu().data.numpy()[0,...]
        print(i)
        if SAVE:
            init_image = transforms.ToPILImage()(denorm_tensor[0]).convert("RGB")
            init_image.save(os.path.join(save_path, f"{videonames}_{str(i).zfill(4)}.png"))