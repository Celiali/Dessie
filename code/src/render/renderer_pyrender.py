import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, center=None, img_w=None, img_h=None, faces=None, same_mesh_color = True):

        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                       viewport_height=img_h,
                                       point_size=1.0)
        self.focal_length = focal_length
        if center is None:
            self.camera_center = [img_w // 2, img_h // 2]
        else:
            self.camera_center = center
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        self.right_index = range(813, 1497)
        self.left_index = range(0, 813)
        self.right_faces = range(1, 2991, 2)
        self.left_faces = range(0, 2990, 2)

    def __call__(self, vertices, image = None, obtainSil = False):
        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        if self.same_mesh_color:
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                alphaMode='OPAQUE',
                baseColorFactor=(0.8, 0.3, 0.3, 1.0))
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        else:
            # mesh.visual.vertex_colors = vertex_colors
            colors_faces = np.zeros_like(mesh.faces)
            colors_faces[self.left_faces, :] = np.array([100, 100, 255]) #np.array([99,110,128])#np.array([136,144,167]) # #np.array([100, 100, 255])
            colors_faces[self.right_faces, :] = np.array([100, 100, 255]) #np.array([99,110,128]) #np.array([136,144,167]) ##np.array([100, 100, 255])
            mesh.visual.face_colors = colors_faces  # np.random.uniform(size=mesh.faces.shape)
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)  # camera_pose[:3, :3] = cam_rot  # camera_pose[:3, 3] = cam_t
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1],
                                           zfar=1000)
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if obtainSil:
            mask = depth_map > 0
            return mask
        elif image is None and not obtainSil:
            return color_rgb
        elif image is not None and not obtainSil:
            mask = depth_map > 0
            image[mask] = color_rgb[mask]*0.7 + image[mask]*0.3
            return image
        else:
            raise ValueError
