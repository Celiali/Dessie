import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,MeshRasterizer, BlendParams,AmbientLights,SoftPhongShader,OpenGLPerspectiveCameras
)
from torchvision import transforms
from src.utils.model_utils import get_point
from src.utils.CONSTANCT import PLABELSNAME

def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel) #, bin_size = 50)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return renderer

def setup_weak_render(image_size, faces_per_pixel,device): #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # render the view
    R = torch.tensor([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]).repeat(1, 1, 1).to(device)
    T = torch.zeros(3).repeat(1, 1).to(device)
    fov = 2 * np.arctan(image_size/ (5000. * 2)) * 180 / np.pi
    cameras = OpenGLPerspectiveCameras(zfar=350, fov=fov, R=R, T=T, device=device)
    renderer = init_renderer(cameras,
                             shader=SoftPhongShader(
                                    cameras=cameras,
                                    lights= AmbientLights(device=device),
                                    device=device,
                                    blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color= (1, 1, 1)),
                                ),
                             image_size=image_size,faces_per_pixel=faces_per_pixel,)
    return cameras, renderer

@torch.no_grad()
def check_visible_verts_batch(mesh, fragments):
    pix_to_face = fragments.pix_to_face
    # (B, F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = mesh._faces_padded #mesh.faces_packed() 
    # (B, V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = mesh._verts_padded #mesh.verts_packed() 
    vertex_visibility_map = torch.zeros((packed_verts.shape[0], packed_verts.shape[1])).to(mesh.device) #[B, V]
    for i in range(packed_verts.shape[0]):
        #  Indices of unique visible faces pix_to_face[i].unique()
        # Get Indices of unique visible verts using the vertex indices in the faces packed_faces[i][pix_to_face[i].unique()]
        visible_per_frame = pix_to_face[i].unique()
        visible_per_frame[1:] -= packed_faces.shape[1]*i
        vertex_visibility_map[i, packed_faces[i][visible_per_frame].reshape(-1,).unique()] =1.
    return vertex_visibility_map

@torch.no_grad()
def render_image_mask_and_save_with_preset_render(renderer, cameras, mesh,image_size,
                                   save_intermediate=False, save_dir = ''):  
    init_images_tensor, fragments = renderer(mesh) # images
    # get mask
    mask_image_tensor = init_images_tensor[...,[-1]].clone()
    mask_image_tensor[mask_image_tensor > 0] = 1.
    # mask_images, _ = mask_renderer(mesh_mask)
    # obtain visible verts
    visible_verts_tensor = check_visible_verts_batch(mesh, fragments) 
    # obtain kps
    kp_3d_tensor = get_point(mesh._verts_padded, flag = 'P', visible_verts = visible_verts_tensor) #[B,17,4]
    kp_2d_tensor = cameras.transform_points_screen(kp_3d_tensor[:,:,:3], image_size=(image_size,image_size))[:,:,:2] #[N, 17, 2]
    kp_2d_tensor = torch.cat([kp_2d_tensor, kp_3d_tensor[:,:,-1].unsqueeze(2)], dim = 2)
     
    B = init_images_tensor.shape[0]         
    if save_intermediate:  
        init_image = init_images_tensor.permute(0,3,1,2).cpu() #[B, 3, I, I]
        init_image = [transforms.ToPILImage()(init_image[t]).convert("RGB") for t in range(B)]
        kp_2d = kp_2d_tensor.cpu()
        # Draw each keypoint as a circle
        for t in range(B):
            # Create a drawing context
            draw = ImageDraw.Draw(init_image[t])
            for tt,keypoint in enumerate(kp_2d[t]):
                # For a circle, we need the top-left and bottom-right coordinates of the bounding square
                x, y, flag = keypoint
                if flag == 1:
                    r = 2  # radius of the circle
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0, 0))
                    # Draw the text onto the image
                    draw.text((x,y), PLABELSNAME[tt], fill= (0, 0, 255) , font= ImageFont.load_default())

        mask_image = mask_image_tensor[:, :, :, 0].cpu() #[B, I, I, 3]
        mask_image = [transforms.ToPILImage()(mask_image[t]).convert("L") for t in range(B)]
        
        # depth_map = [Image.fromarray(depth_maps_tensor.cpu().numpy()[t]).convert("L")  for t in range(B)]
        # save intermediate results
        for t in range(B):
            init_image[t].save(os.path.join(save_dir, "{}_image_old.png".format(t)))
            mask_image[t].save(os.path.join(save_dir, "{}_mask.png".format(t)))
            # depth_map[t].save(os.path.join(save_dir, "{}_depth_map.png".format(t)))
    return (init_images_tensor,mask_image_tensor, kp_3d_tensor, kp_2d_tensor)