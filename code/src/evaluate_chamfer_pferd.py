'''
This is to calculate the chamfer distance for the 3D evaluation. (dinohmr*, dessie*, magicpony)
'''

import numpy as np
import os, trimesh, torch
import pickle as pkl
from pytorch3d.loss.chamfer import chamfer_distance

def get_selected_skeleton_PA(smal_vertices_numpy, J_regressor):
    '''
    smal_vertices: numpy, [1,N,3]
    J_regresor : torch
    point_selected: [1,4,3]
    '''
    # Get joints:
    verts_torch = torch.from_numpy(smal_vertices_numpy).float()
    joint_x = torch.matmul(verts_torch[:, :, 0], J_regressor)
    joint_y = torch.matmul(verts_torch[:, :, 1], J_regressor)
    joint_z = torch.matmul(verts_torch[:, :, 2], J_regressor)
    joints = torch.stack([joint_x, joint_y, joint_z], dim=2).cpu().data.numpy() #[1,36,3]

    point13 = joints[:, [3, 14], :].mean(axis=1) #[1,3]
    point07 = smal_vertices_numpy[:, [127, 57, 565, 65, 940, 870, 56, 878, 41, ], :].mean(axis=1) #122, 869, 600, 854, 935
    point17 = joints[:, 3, :] + (2 / 3) * (
                joints[:, [18, 23], :].mean(axis=1) - joints[:, 3, :])
    point16 = joints[:, [18, 23], :].mean(axis=1)
    point18 = joints[:, 8, :] # left front
    point117 = joints[:,13,:] # right rfront
    point111 = joints[:, 22, :] # left back
    point114 = joints[:, 27, :] # right back
    point10 = joints[:,33,:]
    # point_selected = np.concatenate([point13, point07, point17, point16], axis=0)[None, ...]
    point_selected = np.concatenate([point13, point07, point17, point16,
                                     point18, point117, point111, point114, point10], axis=0)[None, ...]
    return point_selected

def procrustes_transform_ptc(source_points, selected_source_points, target_points, selected_target_points):
    '''
    source_points [1,N,3]
    target_points [1,N,3]
    '''
    if isinstance(source_points, np.ndarray):
        source_points = torch.from_numpy(source_points).float()
        assert len(source_points.shape) == 3
    if isinstance(target_points, np.ndarray):
        target_points = torch.from_numpy(target_points).float()
        assert len(target_points.shape) == 3
    if isinstance(selected_source_points, np.ndarray):
        selected_source_points = torch.from_numpy(selected_source_points).float()
        assert len(selected_source_points.shape) == 3
    if isinstance(selected_target_points, np.ndarray):
        selected_target_points = torch.from_numpy(selected_target_points).float()
        assert len(selected_target_points.shape) == 3
    selected_transformed_source, R, scale, t = compute_similarity_transform(selected_source_points,selected_target_points)
    # Transform S3 using the computed R, scale, and t
    source_transformed = scale * torch.matmul(source_points.clone(), R.permute(0, 2, 1)) + t.permute(0, 2, 1)
    return source_transformed, selected_transformed_source

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) :
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

    return S1_hat.permute(0, 2, 1), R,scale, t

def cal_chamfer_distance(v, v_scan):
    '''
    v [1,N,3]
    v_scan [1,N,3]
    '''
    err, _ = chamfer_distance(v, v_scan)
    return _, _, err

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--SOTA', action="store_true", default=False, help="True: Evaluate with SOTA")
    parser.add_argument('--PONY', action="store_true", default=False, help="True: Calculate magicpony; False: calculate 3dfauna")
    parser.add_argument('--SAVE',action="store_true", default=False, help="save the results")
    parser.add_argument('--PFERD_results', type=str, default='/home/x_cili/x_cili_lic/DESSIE/results/PFERD',)
    parser.add_argument('--model_dir', type=str, default='/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models', help='model dir')
    args = parser.parse_args()
    VISUAL = False
    CHAMFER = True
    
    videodir = [
                '20201128_ID_2_0010_Miqus_65_20715',
                '20201128_ID_2_0008_Miqus_61_23417',
                '20201128_ID_1_0001_Miqus_50_23416',
                '20201129_ID_4_0005_Miqus_64_23414',
                '20201129_ID_4_0008_Miqus_64_23414',
    ]
    cham_sota_error, cham_sota_s2m_error, cham_sota_m2s_error = [],[],[]
    cham_pred_error, cham_pred_s2m_error, cham_pred_m2s_error = [],[],[]
    
    file = 'DESSIE_COMBINAREAL_version_8'
    exp = 'dessie'
        
    PONY = args.PONY 
    if args.SOTA:
        if PONY:
            resultspath = os.path.join(args.PFERD_results, 'magicpony')
            sota_method = 'magicpony'
        else:
            resultspath = os.path.join(args.PFERD_results, '3Dfauna')
            sota_method = '3dfauna'
    else:
        sota_method = ''
    
    SAVE = args.SAVE 
    if SAVE:
        # I want to create a text file to save the results
        textfile = open(os.path.join(args.PFERD_results,f'chamfer_results_{exp}_{sota_method}.txt'), 'w')
    
    print('start evaluation', exp, sota_method)
    # -- Load SMPL params --
    import pickle as pkl
    with open(
            os.path.join(args.model_dir, 'my_smpl_0000_horse_new_skeleton_horse.pkl'),
            'rb') as f:
        dd = pkl.load(f, encoding="latin1")
    hsmal_faces = torch.from_numpy(dd['f'].astype(np.int32)).type(torch.int32).unsqueeze(0)
    J_regressor = torch.from_numpy(dd['J_regressor'].T.todense()).type(torch.float32)
    # rotation x axis 180 degree
    matrix = np.array([[ 1.0,0.,0.],[0.0,-1.0,-0.0],[0.0,0.0,-1.0]])
    
    for v in videodir:    
        path = os.path.join(args.PFERD_results,'Dessie', f'DATA_{v}')
        gt = os.path.join(path, f'{file}_gt.npz')
        pred = os.path.join(path, f'{file}_pred.npz')
        sota3Dpath = os.path.join(resultspath, v) if args.SOTA else None
        # get data
        gt_data = np.load(gt, allow_pickle=True)
        pred_data = np.load(pred, allow_pickle=True)

        frame_all = gt_data['vertices'].shape[0]
        print("current video: ", v)
        for i in range(frame_all):
            frameid = i 

            #### get gt vertices
            gt_vertices = gt_data['vertices'][frameid].copy()  # [1,1497,3]
            # rotate to camera view
            gt_rotated_vertices = np.matmul(gt_vertices, gt_data['R'].T) + gt_data['T']  # Shape: [1, 1479, 3]
            gt_rotated_vertices = np.matmul(gt_rotated_vertices, matrix)
            gt_rotated_vertices_mean = np.mean(gt_rotated_vertices, axis=1).copy()  # [1,1479,3]
            gt_rotated_vertices -= gt_rotated_vertices_mean
            gt_selected_skeleton = get_selected_skeleton_PA(gt_rotated_vertices, J_regressor)

            #### get pred vertices  
            pred_vertices = pred_data['vertices'][frameid].copy()# [1,1479,3]
            # rotate to camera view
            pred_rotated_vertices = np.matmul(pred_vertices, matrix)# [1,1479,3]
            pred_rotated_vertices -= np.mean(pred_rotated_vertices, axis=1)# [1,1479,3]
            pred_selected_skeleton = get_selected_skeleton_PA(pred_rotated_vertices, J_regressor)
            
            #### get_3D sota results
            if args.SOTA:
                if sota_method == '3dfauna':
                    sotaobj = os.path.join(sota3Dpath, f'{v}_{str(frameid).zfill(4)}_reconstructed_shape.obj')
                elif sota_method == 'magicpony':
                    sotaobj = os.path.join(sota3Dpath, f'{v}_{str(frameid).zfill(4)}_mesh.obj')
                sota = trimesh.load(sotaobj)
                sota_vertices = sota.vertices.copy() #[N,3]
                # extra data
                extra_data = np.load(os.path.join(sota3Dpath, f'{v}_{str(frameid).zfill(4)}_shape.npz'), allow_pickle=True)
                w2c = extra_data['w2c'].astype(float)#[1,4,4]
                bones = extra_data['posed_bones'].astype(float) #[1,20,2,3]
                # rotate to camera view
                out = torch.matmul(torch.nn.functional.pad(torch.from_numpy(sota_vertices)[None, ...], pad=(0, 1), mode='constant', value=1.0),torch.transpose(torch.from_numpy(w2c), 1, 2))
                sota_rotated_vertices = (out[..., :3] / out[..., 3:]).cpu().data.numpy() #[1,N,3]
                sota_rotated_vertices_mean = np.mean(sota_rotated_vertices, axis=1).copy()
                sota_rotated_vertices -= sota_rotated_vertices_mean
                # obtain selected skeleton points
                if sota_method == 'magicpony':
                    sota_skeleton = np.concatenate([bones[:, 7, 1, :], bones[:, 7, 0, :], bones[:, 3, 1, :], bones[:, 2, 1, :],
                                                bones[:, 14, 1, :], bones[:, 11, 1, :], bones[:, 17, 1, :], bones[:, 8, 1, :], bones[:,4,1,:]],
                                                axis=0)  
                elif sota_method == '3dfauna':
                    sota_skeleton = np.concatenate([bones[:, 3, 1, :], bones[:, 7, 0, :], bones[:, 7, 1, :], bones[:, 6, 1, :],
                                                    bones[:, 8, 1, :], bones[:, 17, 1, :], bones[:, 11, 1, :], bones[:, 14, 1, :], bones[:,0,1,:]],
                                                    axis=0) 
                sota_rotated_skeleton = torch.matmul(torch.nn.functional.pad(torch.from_numpy(sota_skeleton)[None,...], pad=(0, 1), mode='constant', value=1.0),torch.transpose(torch.from_numpy(w2c), 1, 2))
                sota_rotated_skeleton = sota_rotated_skeleton[..., :3] / sota_rotated_skeleton[..., 3:]
                sota_selected_skeleton = sota_rotated_skeleton.cpu().data.numpy() - sota_rotated_vertices_mean

            # PA
            transformed_prediction, transformed_prediction_selected_skeleton = procrustes_transform_ptc(source_points=pred_rotated_vertices,
                                                                selected_source_points = pred_selected_skeleton,
                                                                target_points=gt_rotated_vertices,
                                                                selected_target_points = gt_selected_skeleton)
            if args.SOTA:
                transformed_sota, transformed_sota_selected_skeleton = procrustes_transform_ptc(source_points=sota_rotated_vertices,
                                                                    selected_source_points = sota_selected_skeleton,
                                                                    target_points=gt_rotated_vertices,
                                                                    selected_target_points = gt_selected_skeleton)

            if CHAMFER:
                ## chamfer distance
                _, _, pred_err_c = cal_chamfer_distance(v = torch.from_numpy(gt_rotated_vertices).float(), v_scan= transformed_prediction)
                if args.SOTA:
                    _, _, sota_err_c = cal_chamfer_distance(v=torch.from_numpy(gt_rotated_vertices).float(), v_scan=transformed_sota) 
                cham_pred_error.append(pred_err_c.item())
                cham_sota_error.append(sota_err_c.item()) if args.SOTA else None
                
        if SAVE:
            start = np.array(cham_pred_error).shape[0] - frame_all
            if CHAMFER:
                textfile.write(f'chamfer, {exp}, {v}, {frame_all}, {np.array(cham_pred_error[start:]).mean()} \n')
                textfile.write(f'chamfer, {sota_method}, {v}, {frame_all}, {np.array(cham_sota_error[start:]).mean()}\n') if args.SOTA else None
                textfile.write(f'\n')                

    if CHAMFER:
        print("model, pred_err scan2mesh mesh2scan")
        print(exp, np.array(cham_pred_error).mean())
        print(sota_method,np.array(cham_sota_error).mean()) if args.SOTA else None

    if SAVE:
        if CHAMFER:
            textfile.write(f'chamfer, {exp},  {np.array(cham_pred_error).mean()} \n')
            textfile.write(f'chamfer, {sota_method}, {np.array(cham_sota_error).mean()} \n') if args.SOTA else None
            textfile.write(f'\n')
        textfile.close()
