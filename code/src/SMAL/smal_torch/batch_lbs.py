from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torch.nn.functional as F

def batch_skew(vec, batch_size=None, device=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
        batch_size = vec.shape.as_list()[0]
    col_inds = torch.LongTensor([1, 2, 3, 5, 6, 7])
    indices = torch.reshape(torch.reshape(torch.arange(0, batch_size) * 9, [-1, 1]) + col_inds, [-1, 1])
    updates = torch.reshape(
            torch.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                dim=1), [-1])
    out_shape = [batch_size * 9]
    res = torch.Tensor(np.zeros(out_shape[0])).to(device)#cuda(device=opts.gpu_id)
    res[np.array(indices.flatten())] = updates
    res = torch.reshape(res, [batch_size, 3, 3])

    return res



def batch_rodrigues(theta, device=None):
    """
    Theta is Nx3
    """
    batch_size = theta.shape[0]

    angle = (torch.norm(theta + 1e-8, p=2, dim=1)).unsqueeze(-1)
    r = (torch.div(theta, angle)).unsqueeze(-1)

    angle = angle.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    outer = torch.matmul(r, r.transpose(1,2))

    eyes = torch.eye(3).unsqueeze(0).repeat([batch_size, 1, 1]).to(device)#cuda(device=opts.gpu_id)
    H = batch_skew(r, batch_size=batch_size, device=device)
    R = cos * eyes + (1 - cos) * outer + sin * H 

    return R

def batch_lrotmin(theta):
    """
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 108 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=36

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 35=(K-1) joints with identity subtracted.,
    """
    # Ignore global rotation
    theta = theta[:,3:]

    Rs = batch_rodrigues(torch.reshape(theta, [-1,3]))
    lrotmin = torch.reshape(Rs - torch.eye(3), [-1, 315])

    return lrotmin

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False, betas_logscale=None, device= None):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 36 x 3 x 3 rotation vector of K joints
      Js: N x 36 x 3, joint locations before posing
      parent: 36 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 36 x 3 location of absolute joints
      A     : `Tensor`: N x 36 4 x 4 relative joint transformations for LBS.
    """
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rot_x = torch.reshape(torch.repeat(rot_x, [N, 1]), [N, 3, 3]) # In tf it was tile
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 36 x 3 x 1
    Js = Js.unsqueeze(-1)
    N = Rs.shape[0]

    # adding scale factor
    Js_orig = Js.clone()
    scaling_factors = torch.ones(N, parent.shape[0], 3).to(device)
    if betas_logscale is not None:
        # code adapt from https://github.com/benjiebob/WLDO/blob/master/global_utils/smal_model/batch_lbs.py
        # front leg list(range(4, 9)): left ;  list(range(9, 14)): right
        # back leg list(range(18, 23)) : left ; list(range(23, 28)): right
        leg_joints = list(range(4, 9)) + list(range(9, 14)) + list(range(18, 23)) + list(range(23, 28))
        tail_joints = list(range(28, 33))
        ear_joints = [34, 35]

        beta_scale_mask = torch.zeros(36, 3, 6).to(device)
        beta_scale_mask[leg_joints, [2], [0]] = 1.0  # Leg lengthening #z
        beta_scale_mask[leg_joints, [0], [1]] = 1.0  # Leg fatness     #x
        beta_scale_mask[leg_joints, [1], [1]] = 1.0  # Leg fatness     #y

        beta_scale_mask[tail_joints, [2], [2]] = 1.0  # Tail lengthening #z
        beta_scale_mask[tail_joints, [0], [3]] = 1.0  # Tail fatness     #x
        beta_scale_mask[tail_joints, [1], [3]] = 1.0  # Tail fatness     #y

        beta_scale_mask[ear_joints, [1], [4]] = 1.0  # Ear y
        beta_scale_mask[ear_joints, [2], [5]] = 1.0  # Ear z

        beta_scale_mask = torch.transpose(
            beta_scale_mask.reshape(36 * 3, 6), 0, 1)

        betas_scale = torch.exp(betas_logscale @ beta_scale_mask)
        scaling_factors = betas_scale.reshape(-1, 36, 3)
    scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)
    # adding scale factor

    def make_A(R, t):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = torch.nn.functional.pad(R, (0,0,0,1,0,0))
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).to(device)], 1) #cuda(device=opts.gpu_id)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        # A_here = make_A(Rs[:, i], j_here)

        # adding scale factor
        rot = Rs[:, i]
        s_par_inv = torch.inverse(scale_factors_3x3[:, parent[i]])
        s = scale_factors_3x3[:, i]
        rot_new = s_par_inv @ rot @ s
        A_here = make_A(rot_new, j_here) #Rs[:, i]
        # adding scale factor
        
        res_here = torch.matmul(
            results[parent[i]], A_here)
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = torch.cat([Js, torch.zeros([N, 36, 1, 1]).to(device)], 2) #.cuda(device=opts.gpu_id)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = F.pad(init_bone, (3,0,0,0,0,0,0,0))
    A = results - init_bone

    return new_J, A
