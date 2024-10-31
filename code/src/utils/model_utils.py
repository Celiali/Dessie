# import matplotlib
# matplotlib.use('TkAgg')
import torch
import numpy as np
from src.utils.CONSTANCT import PPOINTINDEX, ANIMAL3DPOINTINDEX, COMBINAPOINTINDEX

def get_point(verts, flag = 'P', visible_verts= None):
    ##### https://github.com/benjiebob/SMALify/smal_fitter/utils.py
    if flag == 'P':
        SELECTPOINTINDEX = PPOINTINDEX
    elif flag == 'ANIMAL3D':
        SELECTPOINTINDEX = ANIMAL3DPOINTINDEX
    elif flag == 'COMBINAPOINTINDEX':
        SELECTPOINTINDEX = COMBINAPOINTINDEX
    else:
        raise ValueError
    num = verts.shape[0]
    if visible_verts is not  None: # [B, V]
        output = []
        for i in range(num):        
            selected_points = torch.stack(
                    [torch.cat([torch.squeeze(verts[i, choice, :], dim=0), visible_verts[i,choice].any().float().view(-1)], dim=-1)
                    if choice.shape[0] == 1 
                    else 
                    torch.cat([verts[i,choice, :].mean(axis=0), visible_verts[i,choice].any().float().view(-1)], dim=-1)
                    for choice in SELECTPOINTINDEX])
            output.append(torch.unsqueeze(selected_points,dim=0)) #[B,V,4]
    else:
        output =[]
        for i in range(num):
            selected_points = torch.stack(
                [torch.squeeze(verts[i,choice, :],dim=0) if choice.shape[0] == 1 else verts[i,choice, :].mean(axis=0) for choice in SELECTPOINTINDEX])
            output.append(torch.unsqueeze(selected_points,dim=0)) #[B,V,3]
            
    final = torch.cat(output,dim=0).type_as(verts)
    return final

def organize_kp(kp3D, labels):
    # Define the selection indices for each label condition using normal lists
    indices_1_to_4 = list(range(0, 17))  # For labels 1, 2, 3, 4
    indices_5 = [0, 1, 17, 18, 2, 19, 4, 20, 5, 8, 11, 14, 7, 10, 13, 16]

    # Initialize an empty tensor for the output
    B, _, _ = kp3D.shape
    final_kp = torch.empty((B, 17, 3), dtype=kp3D.dtype, device=kp3D.device)

    # Iterate through the batch and select keypoints based on the label
    for i, label in enumerate(labels.squeeze()):
        if label in [1, 2, 3, 4]:  # magicpony label4
            final_kp[i] = kp3D[i, indices_1_to_4]
        elif label == 5: # animal3d label5
            selected_kp = kp3D[i, indices_5]  # Exclude the last duplicated index for selection
            final_kp[i] = torch.cat((selected_kp, torch.zeros((1,3), dtype=kp3D.dtype, device = kp3D.device)), dim=0)  # Duplicate the last kp
        else:
            raise ValueError(f"Unhandled label: {label}")
    return final_kp