import sys, os, torch, cv2, copy,re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.utils.misc import validate_tensor_to_device, validate_tensor, collapseBF
from src.test import get_model
from src.train2 import parse_args
from src.render.renderer_pyrender import Renderer
from PIL import Image
from src.evalpferd_utils.pferd import PFERD
import os.path as osp
import numpy as np
from src.evalpferd_utils.evaluate_utils import Evaluator, EvaluatorPCK
from src.SMAL.smal_torch.smal_torch import SMAL
import pandas as pd
from pathlib import Path
import os

def save_eval_result(
    metric_dict: float,
    checkpoint_path: str,
    dataset_name: str,
    iters_done=None,
    exp_name=None,
    flag = 'mean'
) -> None:
    """Save evaluation results for a single scene file to a common CSV file."""
    version = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path))), f"Eval_PFERD_{version}_eval_results.csv")

    timestamp = pd.Timestamp.now()
    exists: bool = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    # save each metric as different row to the csv path
    metric_names = list(metric_dict.keys())
    metric_values = list(metric_dict.values())
    
    metric_values = [float('{:.2f}'.format(value)) for value in metric_values]
    N = len(metric_names)
    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            iters_done=[iters_done] * N,
            size = [0]*N,
            flag = [flag]*N,
        ),
        index=list(range(N)),
    )

    # Lock the file to prevent multiple processes from writing to it at the same time.
    # lock = FileLock(f"{csv_path}.lock", timeout=10)
    # with lock:
    df.to_csv(csv_path, mode="a", header=not exists, index=False)


KP_INDEX = {
                "C_Poll_1": 586,
                "C_Forehead_2": 573,
                "L_Temple_3": 259,
                "L_Cheek_4": 373,
                "L_Jaw_5": 289,
                "L_Chin_6": 657,
                "R_Temple_3": 1072,
                "R_Cheek_4": 1186,
                "R_Jaw_5": 1102,
                "R_Chin_6": 1381,
                "L_Nostril_In_35": 340,
                "L_Nostril_Out_36": 356,
                "R_Nostril_In_35": 1153,
                "R_Nostril_Out_36": 1169,
                "L_Ear_Base_32": 206,
                "L_Ear_Top_33": 226,
                "L_Ear_Side_34": 221,
                "R_Ear_Base_32": 1019,
                "R_Ear_Top_33": 1039,
                "R_Ear_Side_34": 1034,
                "C_Neck_Top_1_37": 577,
                "C_Neck_Top_2_38": 559,
                "L_Neck_Up_1_41": 50,
                "L_Neck_Up_2_42": 131,
                "L_Neck_Up_3_43": 34,
                "L_Neck_1_7": 22,
                "L_Neck_2_8": 133,
                "L_Neck_3_9": 44,
                "C_Neck_Bottom_1_39": 568,
                "C_Neck_Bottom_2_40": 531,
                "C_Chestpoint_19": 522,
                "R_Neck_Up_1_41": 863,
                "R_Neck_Up_2_42": 944,
                "R_Neck_Up_3_43": 847,
                "R_Neck_1_7": 835,
                "R_Neck_2_8": 946,
                "R_Neck_3_9": 857,
                "L_Shoulder_1_44": 33,
                "L_ShoulderBlade_20": 71,
                "L_ShoulderJoint_21": 70,
                "L_Shoulder_2_45": 69,
                "L_Chest_46": 80,
                "L_ElbowJoint_22": 193,
                "LF_Front_1_47": 6,
                "L_CarpalJoint_23": 9,
                "LF_Front_2_48": 115,
                "LF_FetlockJoint_24": 507,
                "LF_Hoof_Front_71": 485,
                "LF_Hoof_Back_72": 469,
                "LF_Hoof_Side_73": 482,
                "R_Shoulder_1_44": 846,
                "R_ShoulderBlade_20": 884,
                "R_ShoulderJoint_21": 883,
                "R_Shoulder_2_45": 882,
                "R_Chest_46": 893,
                "R_ElbowJoint_22": 1006,
                "RF_Front_1_47": 819,
                "R_CarpalJoint_23": 822,
                "RF_Front_2_48": 928,
                "RF_FetlockJoint_24": 1320,
                "RF_Hoof_Front_71": 1298,
                "RF_Hoof_Back_72": 1282,
                "RF_Hoof_Side_73": 1295,
                "C_Back_2_11": 569,
                "C_Back_3_12": 565,
                "C_Back_4_13": 533,
                "L_Barrel_1_53": 67,
                "L_Barrel_2_54": 58,
                "L_Barrel_3_55": 57,
                "L_Barrel_4_56": 66,
                "L_Barrel_5_57": 121,
                "L_Barrel_6_58": 56,
                "L_Barrel_7_59": 123,
                "L_Barrel_8_60": 1,
                "L_Barrel_9_61": 40,
                "L_Barrel_10_62": 41,
                "L_Barrel_11_63": 42,
                "C_Belly_1_49": 660,
                "C_Belly_2_50": 567,
                "C_Belly_3_51": 600,
                "C_Belly_4_52": 601,
                "R_Barrel_1_53": 880,
                "R_Barrel_2_54": 871,
                "R_Barrel_3_55": 870,
                "R_Barrel_4_56": 879,
                "R_Barrel_5_57": 934,
                "R_Barrel_6_58": 869,
                "R_Barrel_7_59": 936,
                "R_Barrel_8_60": 814,
                "R_Barrel_9_61": 853,
                "R_Barrel_10_62": 854,
                "R_Barrel_11_63": 855,
                "C_Back_5_14": 578,
                "C_Back_6_15": 603,
                "C_Tail_1_16": 756,
                "C_Tail_2_17": 718,
                "C_Tail_3_18": 749,
                "L_Croup_1_64": 25,
                "L_Pelvis_1_25": 126,
                "L_Pelvis_2_26": 156,
                "L_HipJoint_27": 159,
                "L_Thigh_1_65": 157,
                "L_Thigh_2_66": 673,
                "L_StifleJoint_28": 681,
                "L_Croup_2_68": 667,
                "L_Stifle_Back_67": 669,
                "LH_Front_1_69": 15,
                "L_PointOfHock_29": 196,
                "L_HockJoint_30": 51,
                "LH_Front_2_70": 152,
                "LH_FetlockJoint_31": 456,
                "LH_Hoof_Front_74": 446,
                "LH_Hoof_Back_75": 452,
                "LH_Hoof_Side_76": 451,
                "R_Croup_1_64": 838,
                "R_Pelvis_2_26": 969,
                "R_HipJoint_27": 972,
                "R_Thigh_1_65": 970,
                "R_Thigh_2_66": 1395,
                "R_StifleJoint_28": 1403,
                "R_Croup_2_68": 1389,
                "R_Stifle_Back_67": 1391,
                "RH_Front_1_69": 828,
                "R_PointOfHock_29": 1009,
                "R_HockJoint_30": 864,
                "RH_Front_2_70": 965,
                "RH_FetlockJoint_31": 1269,
                "RH_Hoof_Front_74": 1259,
                "RH_Hoof_Back_75": 1265,
                "RH_Hoof_Side_76": 1264,
                "C_Back_1_10": 607,
                "R_Pelvis_1_25": 939
            }

if __name__ == '__main__':
    args = parse_args()
    #####################set data##################################################
    
    videodir = {'20201128_ID_2_0010_Miqus_65_20715': [4,15],
                '20201128_ID_2_0008_Miqus_61_23417': [15,20],
                '20201128_ID_1_0001_Miqus_50_23416': [0,8],
                '20201129_ID_4_0005_Miqus_64_23414': [0,5],
                '20201129_ID_4_0008_Miqus_64_23414': [32,36],
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("========== Preparing model... ========== ")
    Model, trainer = get_model(args, args.ckpt_file, device)
    Model.initial_setup()
    
    # Check the mode right after loading the model
    if Model.training:
        print("Model is in training mode.")
    else:
        print("Model is in evaluation mode.")

    videoskeys = sorted(videodir.keys())
    for i in range(len(videoskeys)):
        videonames = videoskeys[i]
        print(videonames)
        start_time = videodir[videonames][0]
        end_time = videodir[videonames][1]
        
        print("========== Preparing data... ========== ")
        traindataset = PFERD(data_dir= args.data_dir,
                                    videonames = videonames, start_time = start_time, end_time = end_time,)
        train_dl = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=1)

        visualization = False  
        if visualization:
            plt.figure()
            plt.ion()
            
        video = False
        if video:
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint_path))),'PFERD')
            size = (args.imgsize, args.imgsize)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(save_path,
                            f'{videonames}_{args.ModelName}_{args.name}_version_{os.path.basename(os.path.dirname(os.path.dirname(args.ckpt_file)))}.mp4'), fourcc, traindataset.videoFps, size)   

        print("========== Preparing render... ========== ")
        ####setup render####
        faces_cpu = Model.smal_model.faces.unsqueeze(0).cpu().data.numpy()[0]
        render = Renderer(focal_length=5000, img_w=args.imgsize, img_h=args.imgsize, faces=faces_cpu, same_mesh_color=False)    

        print("=========== setup evaluator ===========")
        Evaluation3D = True
        # Setup evaluator object
        if Evaluation3D:
            evaluator = Evaluator(
                dataset_length=int(1e8), 
                metrics=['mode_mpjpe_rigid', 'mode_re_rigid',],    
            )
        
        Evaluation2D = True
        if Evaluation2D:
            evaluator2Dpck = EvaluatorPCK(
                dataset_length=int(1e8), 
                metrics=['mode_rigid_pck10'],    
            )
        
        print("===========  initial hSMAL model ===========")
        smal_model = SMAL(os.path.join(args.model_dir, 'my_smpl_0000_horse_new_skeleton_horse.pkl'),device=device, use_smal_betas = True)
        
        print("=========== get kp index ===========")
        pattern = re.compile(r".+_.+_(\d+)$")
        # Extract indices where the key matches the pattern and the number is between 1 and 31
        rigid_indices = [index for key, index in KP_INDEX.items() if pattern.match(key) and 1 <= int(pattern.match(key).group(1)) <= 31]
        # Extract indices that do not match the pattern or the number is not between 1 and 31
        rest_indices = [index for key, index in KP_INDEX.items() if not (pattern.match(key) and 1 <= int(pattern.match(key).group(1)) <= 31)]
        
        # get the order of the keypoints
        rigid_name = [ traindataset.kplabels[t] for t in traindataset.rigidmarker]
        rest_name = [traindataset.kplabels[t] for t in traindataset.restmarker]
        rigid_order = [ KP_INDEX[i]  for i in rigid_name]  # get the order of the keypoints in the SMAL model
        rest_order = [ KP_INDEX[i]  for i in rest_name] # get the order of the keypoints in the SMAL model
        
        assert sorted(rigid_indices) == sorted(rigid_order)
        assert sorted(rest_indices) == sorted(rest_order)
        
        print("=========== start evaluation ===========")
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
                pred_rigid_kp = pred_vertices[:, rigid_indices]
                pred_rest_kp = pred_vertices[:, rest_indices]
                
                ### get predicted 2D kp
                pred_rigid_3Dkp = pred_vertices[:, rigid_order, :]
                pred_rest_3Dkp = pred_vertices[:, rest_order, :]
                _, pred_rigid_p2d = Model.get_render_results(verts=pred_vertices, faces=Model.faces,points=pred_rigid_3Dkp)
                _, pred_rest_p2d = Model.get_render_results(verts=pred_vertices, faces=Model.faces,points=pred_rest_3Dkp)
                
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
                gt_rigid_kp = gt_rotated_vertices[:, rigid_indices]
                gt_rest_kp = gt_rotated_vertices[:, rest_indices]
                ### get gt 2D kp
                gt_2D =  collapseBF(data[11]).to(device) #[1,1,132,3] --> [1,132,3]
                gt_rigid_p2d = gt_2D[:, traindataset.rigidmarker, :] # kp index 
                gt_rest_p2d = gt_2D[:, traindataset.restmarker, :] # kp index

                # evaluation
                if Evaluation3D:
                    evaluator(output = {"pred_joint":pred_joint, "pred_rigid_kp": pred_rigid_kp, },
                        batch = {"gt_joint":gt_rotated_joint, "gt_rigid_kp": gt_rigid_kp, })
            
                # evaluation
                if Evaluation2D:
                    evaluator2Dpck(output = {'pred_rigid_p2d':pred_rigid_p2d,},
                            batch = {'gt_rigid_p2d':gt_rigid_p2d, })
                
                ### save video
                if video:
                    original_image = denormalized_image_tensor[0].cpu().data.numpy().transpose(1,2,0)*255
                    overlap_image = render(pred_vertices[0].cpu().data.numpy(), image = original_image, obtainSil = False)
                    color_image_Image = Image.fromarray((overlap_image).astype(np.uint8)).convert('RGB')
                    color_image_numpy = np.array(color_image_Image)[:,:,::-1]
                    videoWriter.write(color_image_numpy)

                if i % 500 == 0:
                    evaluator.log() if Evaluation3D else None
                    evaluator2Dpck.log() if Evaluation2D else None
                    if visualization:
                        original_image = denormalized_image_tensor[0].cpu().data.numpy().transpose(1,2,0)*255
                        overlap_image = render(pred_data['pred_vertices_crop'][0].cpu().data.numpy(), image = original_image, obtainSil = False)
                        color_image_Image = Image.fromarray((overlap_image).astype(np.uint8)).convert('RGB')
                        color_image_numpy = np.array(color_image_Image)[:,:,::-1]
                        plt.cla()
                        plt.imshow(color_image_Image)
                        plt.title(f'img:{i}')
                        plt.show()
                        plt.pause(0.5)
                        plt.axis('off')
                            
            evaluator.log() if Evaluation3D else None
            evaluator2Dpck.log() if Evaluation2D else None
            print(args.name, args.ckpt_file)
            # Append results to file
            if Evaluation3D:
                metrics_dict_mean, metrics_dict_std, metrics_dict_median= evaluator.get_metrics_dict()
                print(metrics_dict_mean)
            if Evaluation2D:
                pck_metrics_dict_mean, pck_metrics_dict_std, pck_metrics_dict_median= evaluator2Dpck.get_metrics_dict()
                print(pck_metrics_dict_mean)
            
            if Evaluation3D:
                save_eval_result( metrics_dict_mean, args.ckpt_file, videonames, iters_done=i, exp_name=args.name, flag = 'mean')
            if Evaluation2D:
                save_eval_result( pck_metrics_dict_mean, args.ckpt_file, videonames, iters_done=i, exp_name=args.name, flag = 'pck_mean')
            
            if video:
                videoWriter.release()
                print(f"saved video {os.path.join(save_path,f'{videonames}_{args.ModelName}_{args.name}_version_{os.path.basename(os.path.dirname(os.path.dirname(args.ckpt_file)))}.mp4')}")