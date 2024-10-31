"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl 
from src.SMAL.smal_torch.batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from src.SMAL.smal_torch.smal_basics import align_smal_template_to_symmetry_axis

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(object):
    def __init__(self, model_path, device, use_smal_betas=False, dtype=torch.float):
        self.device = device 
        self.use_smal_betas = use_smal_betas
        # -- Load SMPL params --
        with open(model_path, 'rb') as f:
            dd = pkl.load(f, encoding="latin1")

        self.faces = torch.from_numpy(dd['f'].astype(np.int32)).type(torch.int32).to(self.device)

        v_template = dd['v_template']
        v, self.left_inds, self.right_inds, self.center_inds, symIdx = align_smal_template_to_symmetry_axis(model_path, v_template)

        # symIdx
        self.symIdx = Variable(
            torch.tensor(symIdx).long().to(self.device), 
            requires_grad=False)

        # Mean template vertices
        self.v_template = Variable(
            torch.from_numpy(v).type(torch.float32).to(self.device), 
            requires_grad=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis

        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = Variable(
            torch.from_numpy(shapedir.copy()).type(torch.float32).to(self.device),requires_grad=False)

        # Regressor for joint locations given shape
        self.J_regressor = Variable(
            torch.from_numpy(dd['J_regressor'].T.todense()).type(torch.float32).to(self.device), 
            requires_grad=False)

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]

        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Variable(
            torch.from_numpy(posedirs.copy()).type(torch.float32).to(self.device),requires_grad=False) 

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(
            torch.from_numpy(undo_chumpy(dd['weights'])).type(torch.float32).to(self.device), 
            requires_grad=False)

    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True):

        if self.use_smal_betas: 
            nBetas = beta.shape[1]
        else:
            nBetas = 9

        # 1. Add shape blend shapes

        if nBetas > 0:
            if del_v is None:
                v_shaped = self.v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = self.v_template.unsqueeze(0)
            else:
                v_shaped = self.v_template + del_v

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 36 x 3 x 3
        if len(theta.shape) ==4:
            Rs = theta
        else:
            Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3]), device=self.device), [-1, 36, 3, 3])
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(self.device),  [-1, 315])



        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        #4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, device=self.device, betas_logscale=betas_logscale)


        # 5. Do skinning:
        num_batch = theta.shape[0]

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 36])


        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 36, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device = self.device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device = self.device) 

        verts = verts + trans[:,None,:]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def test_gpu(opts):
    from time import time
    import os
    if opts.gpu != 'cpu' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print(device)

    pose_size = 108
    beta_size = 9

    np.random.seed(9608)
    print(os.path.abspath(__file__))
    model = SMAL(model_path='../smpl_models/my_smpl_0000_horse_new_skeleton_horse.pkl', device=device, use_smal_betas=False)
    for i in range(10):
        pose = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 0.4) \
            .type(torch.float32).to(device)
        # pose = torch.from_numpy(np.zeros((1, pose_size))).type(torch.float32).to(device)
        betas = torch.from_numpy((np.random.rand(32, beta_size) - 0.5) * 0.06) \
            .type(torch.float32).to(device)
        s = time()
        trans = torch.from_numpy(np.zeros((32, 3))).type(torch.float32).to(device)
        result, joints, Rs = model(betas, pose, trans)
        print(time() - s)
    
if __name__ == '__main__':
    import sys, os

    print(sys.path)
    print(__file__)
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(base_path)
    print(sys.path)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import argparse
    parser = argparse.ArgumentParser(description='Base Processor')
    parser.add_argument('--gpu', default='cpu', help='gpu id')
    parser.add_argument('--use_smal_betas', default=False, help='if using smal shape space')
    opts = parser.parse_args()
    test_gpu(opts)