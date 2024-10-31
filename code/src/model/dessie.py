# This script is borrowed and extended from SPIN

import torch
import torch.nn as nn
import numpy as np
import math, os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))
from src.utils.geometry import rot6d_to_rotmat
from src.model.extractor import VitExtractor

class DESSIE(nn.Module): #Text2FCDINODisenNewHMR
    def __init__(self, opts, ):
        super(DESSIE, self).__init__()
        npose = 36 * 6
        nshape = 9
        self.pred_trans = opts.pred_trans
        if self.pred_trans: # we use pred_trans = True
            ncam = 3
        else:
            ncam = 2  

        print("DESSIE")
        self.obtain_token = opts.DINO_obtain_token
        nbbox = 3
        self.imgsize = opts.imgsize
        if opts.DINO_model_name == 'dino_vits8':
            img_feat_num = 384
            kernel_size = 7 if self.imgsize == 224 else 8
        else:
            raise ValueError
        
        if kernel_size is None:
            raise NotImplementedError

        self.vit_feat_dim = img_feat_num
        text_embedding = 640
        fc1_feat_num = 1024
        fc2_feat_num = text_embedding
        final_feat_num = fc2_feat_num

        if self.obtain_token:
            raise NotImplementedError
        else: # we use the key features
            self.key_encoder_shape = nn.Sequential(
                nn.Conv2d(img_feat_num, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2), # Output size: [2, 256, 16, 16]
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2), # Output size: [2, 128, 8, 8]
                nn.Conv2d(128, text_embedding, kernel_size=kernel_size)  #  # Output size: [2, 640, 1, 1]
            )
            self.key_encoder_pose = nn.Sequential(
                nn.Conv2d(img_feat_num, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 256, 16, 16]
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 128, 8, 8]
                nn.Conv2d(128, text_embedding, kernel_size=kernel_size)  # # Output size: [2, 640, 1, 1]
            )
            self.key_encoder_cam = nn.Sequential(
                nn.Conv2d(img_feat_num, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 256, 16, 16]
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: [2, 128, 8, 8]
                nn.Conv2d(128, text_embedding, kernel_size=kernel_size)  # # Output size: [2, 640, 1, 1]
            )

        self.drop1 = nn.Dropout()
        self.fc_decpose1 = nn.Linear(final_feat_num + 35 * 6, fc1_feat_num)
        self.fc_decpose2 = nn.Linear(fc1_feat_num, final_feat_num)
        self.decpose = nn.Linear(final_feat_num, 35 * 6)

        self.fc_decshape1 = nn.Linear(final_feat_num + nshape, fc1_feat_num)
        self.fc_decshape2 = nn.Linear( fc1_feat_num, final_feat_num)
        self.decshape = nn.Linear(final_feat_num, nshape)

        self.fc_deccam1 = nn.Linear(final_feat_num + ncam + 6, fc1_feat_num)
        self.fc_deccam2 = nn.Linear(fc1_feat_num, final_feat_num)
        self.deccam = nn.Linear(final_feat_num, ncam)
        self.decroot = nn.Linear(final_feat_num, 1 * 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

        self.encoder = VitExtractor(model_name=opts.DINO_model_name, frozen = opts.DINO_frozen)

        mean_params = np.load(os.path.join(opts.model_dir, '6D_meanpose.npz'), allow_pickle=True)['mean_pose'].astype(
            np.float32)
        mean_params[0, :] = [-0.0517, 0.9979, -0.2873, -0.0517, -0.9564, -0.0384]
        init_pose = torch.reshape(torch.from_numpy(mean_params.copy()).unsqueeze(0), [1, -1])
        init_shape = torch.from_numpy(np.zeros(9)).unsqueeze(0).type(torch.float32)
        if self.pred_trans:
            init_cam = torch.from_numpy(np.array([0.6,0.0,0.0])).unsqueeze(0).type(torch.float32)
        else:
            init_cam = torch.from_numpy(np.array([0.0, 0.0])).unsqueeze(0).type(torch.float32)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def obtain_latent_space(self, x):
        if self.obtain_token :
            raise NotImplementedError
        else:
            ps = self.encoder.get_patch_size()
            pw,ph = self.imgsize // ps, self.imgsize // ps

            b = x.shape[0]
            xf = self.encoder.get_keys_from_input(x, layer_num=11)[:,:,1:,:].permute(0, 1, 3, 2).reshape(b, self.vit_feat_dim, ph, pw)
            xf_shape = self.key_encoder_shape(xf).view(b, -1)
            xf_pose = self.key_encoder_pose(xf).view(b, -1)
            xf_cam = self.key_encoder_cam(xf).view(b, -1)
        return xf_shape, xf_pose, xf_cam

    def predict_shape(self, x, init_shape):
        pred_shape = init_shape
        for i in range(3):
            xc = torch.cat([x, pred_shape], 1)
            xc = self.fc_decshape1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decshape2(xc)
            xc = self.drop1(xc)
            pred_shape = self.decshape(xc) + pred_shape
        return pred_shape

    def predict_pose(self, x, init_pose):
        pred_pose = init_pose
        for i in range(3):
            xc = torch.cat([x, pred_pose], 1)
            xc = self.fc_decpose1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decpose2(xc)
            xc = self.drop1(xc)
            pred_pose = self.decpose(xc) + pred_pose
        return pred_pose  # , pred_rotmat

    def predict_cam(self, x, init_cam, init_root):
        pred_cam = init_cam
        pred_root = init_root
        for i in range(3):
            xc = torch.cat([x, pred_cam, pred_root], 1)
            xc = self.fc_deccam1(xc)
            xc = self.drop1(xc)
            xc = self.fc_deccam2(xc)
            xc = self.drop1(xc)
            pred_cam = self.deccam(xc) + pred_cam
            pred_root = self.decroot(xc) + pred_root
        return pred_cam, pred_root

    def predict_from_latent(self, xf_shape, xf_pose, xf_cam):
        # predict
        batch_size = xf_shape.shape[0]
        pred_shape = self.predict_shape(xf_shape, init_shape=self.init_shape.expand(batch_size, -1))
        pred_pose = self.predict_pose(xf_pose, init_pose=self.init_pose[0, 6:].expand(batch_size, -1))
        pred_cam, pred_root = self.predict_cam(xf_cam, init_cam=self.init_cam.expand(batch_size, -1),
                                               init_root=self.init_pose[0, :6].expand(batch_size, -1))
        pred_6D = torch.cat([pred_root, pred_pose], 1)
        pred_rotmat = rot6d_to_rotmat(pred_6D).view(batch_size, 36, 3, 3)
        if not self.pred_trans:
            pred_cam = torch.cat([torch.tensor([[0.6]]).repeat(batch_size, 1).float().to(pred_pose.device), pred_cam],dim=1)
        return pred_rotmat, pred_shape, pred_cam

    def forward(self, x, n_iter=3):
        xf_shape, xf_pose, xf_cam = self.obtain_latent_space(x)
        pred_rotmat, pred_shape, pred_cam = self.predict_from_latent(xf_shape, xf_pose, xf_cam)
        return pred_rotmat, pred_shape, pred_cam, xf_shape, xf_pose, xf_cam

def test_DESSIE():
    class Options:
        def __init__(self, model_dir, model_name, frozen, obtain_token, imgsize,pred_trans):
            self.model_dir = model_dir
            self.DINO_model_name = model_name
            self.DINO_frozen = frozen
            self.DINO_obtain_token = obtain_token
            self.imgsize = imgsize
            self.pred_trans = pred_trans

    # Usage:
    opts = Options(model_dir="/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models",
                   model_name = 'dino_vits8',frozen = True, obtain_token = False, imgsize = 256, pred_trans = True
    )
    inp = torch.rand(2, 3, 256, 256)
    model = DESSIE(opts)
    out = model(inp, )
    print('rot', out[0].shape, 'shape', out[1].shape, 'cam', out[2].shape)

if __name__ == '__main__':
    test_DESSIE()