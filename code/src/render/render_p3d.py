import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, SoftSilhouetteShader,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,PerspectiveCameras, HardPhongShader,PointLights,TexturesVertex
)


class base_renderer():
    def __init__(self, size, focal=None, fov=None, device='cpu', T = None, colorRender = False):
        self.device = device
        self.size = size

        self.R = torch.tensor([[-1, 0, 0],
                               [0, -1, 0],
                               [0, 0, 1]]).repeat(1, 1, 1).to(device)
        if T is not None:
            self.T = T.to(device)
        else:
            self.T = torch.zeros(3).repeat(1, 1).to(device)

        self.camera = self.init_camera(focal, fov)
        self.silhouette_renderer = self.init_silhouette_renderer()
        if colorRender:
            self.color_render = self.color()
        else:
            self.color_render = None

    def init_camera(self, focal, fov):
        if fov is None:
            fov = 2 * np.arctan(self.size / (focal * 2)) * 180 / np.pi

        camera = OpenGLPerspectiveCameras(zfar=350, fov=fov, R = self.R, T = self.T, device=self.device)

        #not work camera = PerspectiveCameras(focal_length=[[focal,focal],],image_size=[[self.size,self.size],],device=self.device, R=self.R, T=self.T)
        return camera

    def init_silhouette_renderer(self):
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
            perspective_correct=False,
        )

        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        return silhouette_renderer

    def color(self):
        raster_settings_color = RasterizationSettings(
            image_size=self.size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
        )
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        color_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings_color
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=lights,
            )
        )
        return color_renderer

    def get_color_image(self, vertices, faces, model = None):
        '''  render color image
            Input:
            vertices: BN * V * 3
            faces: BN * F * 3
        '''
        if self.color_render is None:
            raise ValueError
        if model is not None:
            torch_mesh = model
        else:            
            tex = torch.ones_like(vertices)  # (1, V, 3)
            textures = TexturesVertex(verts_features=tex)
            torch_mesh = Meshes(verts=vertices.to(self.device),
                                faces=faces.to(self.device),textures= textures.to(self.device))
        color_image = self.color_render(torch_mesh).permute(0, 3, 1, 2)[:, :3, :, :]
        return color_image
    
    def __call__(self, vertices, faces, points = None):
        ''' Right now only render silhouettes
            Input:
            vertices: BN * V * 3
            faces: BN * F * 3
            points: BN * V * 3
        '''
        torch_mesh = Meshes(verts=vertices.to(self.device),
                            faces=faces.to(self.device))
        
        silhouette = self.silhouette_renderer(meshes_world=torch_mesh.clone(),
                                              R=self.R, T=self.T)#[..., -1]

        screen_size = torch.ones(1, 2).to(self.device) * self.size #torch.ones(vertices.shape[0],2)
        if points is not None:
            proj_points = self.camera.transform_points_screen(points.to(self.device), image_size=screen_size)[:, :, :2]

        if points is not None:
            return silhouette, proj_points
        else:
            return silhouette
