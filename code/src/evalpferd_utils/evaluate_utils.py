"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"https://github.com/saidwivedi/TokenHMR/blob/main/tokenhmr/eval.py"
"""
import cv2
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    # Ensure that the input is of type float32
    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstruction_error
    r_error = reconstruction_error(pred_joints, gt_joints).cpu().numpy()
    return 1000 * mpjpe, 1000 * r_error

class Evaluator:

    def __init__(self,
                 dataset_length: int,
                 metrics: List = ["mode_mpjpe_rigid","mode_re_rigid"],
                 dataset=''):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.pelvis_ind = 0
        self.metrics = metrics
        self.dataset = dataset
        for metric in self.metrics:
            setattr(self, metric, np.zeros((dataset_length,)))
        self.counter = 0

        self.imgnames = []

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} / {self.dataset_length} samples')
        for metric in self.metrics:
            if metric in ['mode_mpjpe_rigid', 'mode_re_rigid']:
                unit = 'mm'
            else:
                unit = ''
            print(f'{metric}: {getattr(self, metric)[:self.counter].mean(0)} {unit}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        d1 = {metric: getattr(self, metric)[:self.counter].mean() for metric in self.metrics}
        d2 = {metric: getattr(self, metric)[:self.counter].std() for metric in self.metrics}
        d3 = {metric: np.median(getattr(self, metric)[:self.counter]) for metric in self.metrics}
        return d1, d2, d3
    
    def get_imgnames(self):
        return self.imgnames

    def __call__(self, output: Dict, batch: Dict):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
        """
        pred_keypoints_3d = output['pred_joint'].detach()[:,None,:,:] #[1,36,3] -> [1,1,36,3]
        pred_rigid_joint = output['pred_rigid_kp'].detach()[:,None,:,:]
        
        gt_keypoints_3d = batch['gt_joint'].detach()[:, None,:,: ] # #[1,1,36,3]
        gt_rigid_joint = batch['gt_rigid_kp'].detach()[:,None,:,:]
        
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        
        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_pelvis = pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_pelvis = gt_keypoints_3d[:, :, [self.pelvis_ind]]
        
        #kp3d
        pred_keypoints_3d -= pred_pelvis
        gt_keypoints_3d -= gt_pelvis
        pred_keypoints_3d = pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3) #[T, N, 3 ]
        gt_keypoints_3d =  gt_keypoints_3d.reshape(batch_size * num_samples, -1 ,3)
        #kp rigid
        pred_rigid_joint -= pred_pelvis
        gt_rigid_joint -= gt_pelvis
        pred_rigid_joint = pred_rigid_joint.reshape(batch_size * num_samples, -1, 3)
        gt_rigid_joint = gt_rigid_joint.reshape(batch_size * num_samples, -1, 3)
        
        mpjpe_rigid, re_rigid = eval_pose(pred_rigid_joint, gt_rigid_joint)
        mpjpe_rigid = mpjpe_rigid.reshape(batch_size, num_samples)
        re_rigid = re_rigid.reshape(batch_size, num_samples)

        if hasattr(self, 'mode_mpjpe_rigid'):
            mode_mpjpe_rigid = mpjpe_rigid[:, 0]
            self.mode_mpjpe_rigid[self.counter:self.counter+batch_size] = mode_mpjpe_rigid
        if hasattr(self, 'mode_re_rigid'):
            mode_re_rigid = re_rigid[:, 0]
            self.mode_re_rigid[self.counter:self.counter+batch_size] = mode_re_rigid

        self.counter += batch_size

        if hasattr(self, 'mode_mpjpe_rigid') and hasattr(self, 'mode_re_rigid'):
            return {
                'mode_mpjpe_rigid': mode_mpjpe_rigid,
                'mode_re_rigid': mode_re_rigid,
            }
            
class EvaluatorPCK:

    def __init__(self,
                 dataset_length: int,
                 metrics: List = ['mode_rigid_pck10'],
                 dataset=''):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.metrics = metrics
        self.dataset = dataset
        for metric in self.metrics:
            setattr(self, metric, np.zeros((dataset_length,)))
        self.counter = 0

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} / {self.dataset_length} samples')
        for metric in self.metrics:
            if metric in ['mode_rigid_pck10']:
                unit = ''
            else:
                unit = ''
            print(f'{metric}: {getattr(self, metric)[:self.counter].mean(0)}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        d1 = {metric: getattr(self, metric)[:self.counter].mean() for metric in self.metrics}
        d2 = {metric: getattr(self, metric)[:self.counter].std() for metric in self.metrics}
        d3 = {metric: np.median(getattr(self, metric)[:self.counter]) for metric in self.metrics}
        return d1, d2, d3
    
    def get_imgnames(self):
        return self.imgnames
    
    def get_pck(self, proj_kpts, gt_kpts):
        assert proj_kpts.shape[0] == gt_kpts.shape[0] == 1
        err = proj_kpts[:, :, :2] - gt_kpts[:, :, :2] #[1,N,2]
        err = err.norm(dim=2, keepdim=True) #[1,N,1]

        maxHW = 256
        i = 0
        valid = gt_kpts[i, :, 2:] > 0
        err_i = err[i][valid]
        err_i = err_i / maxHW
        pck01 = (err_i < 0.01).float().mean().item()
        pck05 = (err_i < 0.05).float().mean().item()
        pck10 = (err_i < 0.10).float().mean().item()
        return pck01, pck05, pck10
            

    def __call__(self, output: Dict, batch: Dict):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
        """
        pred_rigid_keypoints = output['pred_rigid_p2d'].detach()#[1,36,3] 
        gt_rigid_keypoints = batch['gt_rigid_p2d'].detach() #[1,36,3] 
        
        
        rigid_pck01, rigid_pck05, rigid_pck10 = self.get_pck(pred_rigid_keypoints, gt_rigid_keypoints)
        
        batch_size = pred_rigid_keypoints.shape[0]
        # The 0-th sample always corresponds to the mode
        if hasattr(self, 'mode_rigid_pck10'):
            self.mode_rigid_pck10[self.counter:self.counter+batch_size] = rigid_pck10
        self.counter += batch_size

        if hasattr(self, 'mode_rigid_pck10'):
            return {
                'mode_rigid_pck10': rigid_pck10,
            }