import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))
import torch, os, glob, json, imageio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFont
import numpy as np
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.structures import Meshes
import os.path as osp

from src.utils.data_utils import render_image_mask_and_save_with_preset_render, setup_weak_render
from src.SMAL.smal_torch.smal_torch import SMAL
from src.utils.CONSTANCT import TEXTURE_Neworder, QUAD_JOINT_PERM,QUAD_JOINT_NAMES,JOINT_PERM,PLABELSNAME
import gc
from pytorch3d.io import load_obj
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from src.utils.geometry import rotmat_to_axis_angle

def none_to_nan(x):
    if x is None:
        return torch.FloatTensor([float('nan')])
    elif isinstance(x, int):
        return torch.FloatTensor([x])
    else:
        return x
    # return torch.FloatTensor([float('nan')]) if x is None else x


# Define a custom transform function
def convert_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img


class DessiePIPEWithRealImage(Dataset):
    def __init__(self, args, device, length, FLAG, cameraindex=None):
        """
        Args:
            generate_data_function (callable): Function to generate data.
            length (int): Length of the dataset.
        """
        print(f'dataset: DessiePIPEWithRealImage: {FLAG}')
        self.length = length
        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])
        self.FLAG = FLAG
        self.args = args
        self.device = device
        self.image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        self.image_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        self.transform = True
        self.get_data()

    def process_image(self, image_tensor):
        img_process = torch.zeros_like(image_tensor)
        for i in range(img_process.shape[0]):
            img = transforms.ToPILImage()(image_tensor[i, ...])
            img = convert_rgb(img)
            if self.FLAG == 'TEST':
                img = self.image_transform2(img)
            else:
                img = self.image_transform(img)
            img_process[i] = img
        return img_process

    def get_data(self):
        self.get_model()
        self.get_vt_ft()
        self.get_predefinedata()
        self.get_texture()
        self.get_rotation_angle()
        self.get_background()
        self.get_real_image_list()
    
    def get_animal3d_train_list(self):
        filelist = []
        annos = []
        anno_file = os.path.join(self.args.REALPATH, f'coco_pascal/annotations/coco_pascal_nomask.jsons') 
        with open(anno_file) as f:
            anno = json.load(f)
        # load filelist
        pascal_filelist = sorted([str(ann['img_id']) for ann in anno])
        num_imgs = len(pascal_filelist)
        print('Pascal: {} training images'.format(num_imgs))
        for ann in anno:
            ann['image_path'] = os.path.join(self.args.REALPATH, f'coco_pascal/images', os.path.basename(ann['image_path']) )
        annos    += anno
        filelist += pascal_filelist
        return filelist, annos

    def get_animal3d_val_list(self):
        # load annoatations
        anno_file = os.path.join(self.args.REALPATH, f'pascal_val/annotations/pascal_val.jsons')
        with open(anno_file) as f:
            anno = json.load(f)
        # load filelist
        pascal_filelist = sorted([str(ann['img_id']) for ann in anno])
        num_imgs = len(pascal_filelist)
        print('pascal: {} valid images'.format(num_imgs))
        for ann in anno:
            ann['image_path'] = os.path.join(self.args.REALPATH, f'pascal_val/images', os.path.basename(ann['image_path']) )
        return pascal_filelist, anno

    def get_magicpony_train_list(self):
        trainfilelist = sorted(glob.glob(os.path.join(self.args.REALMagicPonyPATH,'train', '**/*_rgb.jpg'), recursive=True))
        validfilelist = sorted(glob.glob(os.path.join(self.args.REALMagicPonyPATH,'val', '**/*_rgb.jpg'), recursive=True))
        print('magicpony: {} train images'.format(len(trainfilelist)))
        print('magicpony: {} valid images'.format(len(validfilelist)))
        return trainfilelist, validfilelist        

    def get_real_image_list(self):
        if self.args.REALDATASET == 'Animal3D': # Staths dataset
            filelist, annos = self.get_animal3d_train_list()
            self.animal3drealimg_filelist_train  = filelist
            self.animal3drealanno_dict_train = {str(anno['img_id']): anno for anno in annos}
            print(f'=> Staths {len(self.animal3drealimg_filelist_train)} images it total')
            filelist, annos = self.get_animal3d_val_list()
            self.animal3drealimg_filelist_eval  = filelist
            self.animal3drealanno_dict_eval = {str(anno['img_id']): anno for anno in annos}
            print(f'=> Staths {len(self.animal3drealimg_filelist_eval)} images it total')
            self.realimg_filelist_train = self.animal3drealimg_filelist_train
            self.realimg_filelist_eval = self.animal3drealimg_filelist_eval
        elif self.args.REALDATASET == 'MagicPony':
            self.magicponytrainfilelist, self.magicponyvalidfilelist = self.get_magicpony_train_list()
            print(f'=> MagicPony {len(self.magicponytrainfilelist)} images it total')
            print(f'=> MagicPony {len(self.magicponyvalidfilelist)} images it total')
            self.realimg_filelist_train = self.magicponytrainfilelist
            self.realimg_filelist_eval = self.magicponyvalidfilelist
        else:
            raise ValueError(f'Unknown REALDATASET: {self.args.REALDATASET}')

    def get_background(self):
        if not hasattr(self, "background_train"):
            self.background_train = sorted(glob.glob(os.path.join(self.args.background_path, 'train2017', '*.jpg')))
            self.background_trainingindex = [i for i in range(len(self.background_train))]
            self.background_valid = sorted(glob.glob(os.path.join(self.args.background_path, 'val2017', '*.jpg')))
            self.background_validindex = [i for i in range(len(self.background_valid))] 
            self.background_test = sorted(glob.glob(os.path.join(self.args.background_path, 'test2017', '*.jpg')))
            self.background_testdindex = [i for i in range(len(self.background_test))]
            self.background_transform = transforms.Compose([transforms.Resize((self.args.imgsize, self.args.imgsize)),transforms.ToTensor()])

    def get_model(self):
        # load model
        self.smal_model = SMAL(os.path.join(self.args.model_dir, 'my_smpl_0000_horse_new_skeleton_horse.pkl'),
                               device=self.device)
        self.faces = self.smal_model.faces.unsqueeze(0)

    def get_vt_ft(self):
        # load for texture
        if not hasattr(self, "ft") or not hasattr(self, "vt"):
            obj_filename = os.path.join(os.path.dirname(self.args.model_dir),
                                        "uvmap/uvmap_from_TextPaper/TexPaper_uv_mask.obj")
            smpl_texture_data = load_obj(obj_filename)
            self.vt = smpl_texture_data[-1].verts_uvs
            self.ft = smpl_texture_data[1].textures_idx

    def get_predefinedata(self):
        # load data
        pose1 = np.load(os.path.join(self.args.PosePath, 'poses1.npz'), allow_pickle=True)['poses']
        pose2 = np.load(os.path.join(self.args.PosePath, 'poses2.npz'), allow_pickle=True)['poses'][:12720, :]  # whole
        '''
        # Define the ranges and the corresponding number of values to sample and # Perform the sampling for each range
        sampling_ranges = [(0, 663, 132), (663, 1275, 122),  (1275, 1459, 36),  (1459, 1590, 26)]
        random samples some data from the whole data for testing
        '''
        testindex_interval = [491, 44, 546, 383, 33, 363, 272, 640, 450, 481, 123, 58, 86, 329, 499, 555, 444, 382, 198,
                        193, 515, 566, 430, 378, 642, 288, 434, 605, 99, 467, 645, 274, 498, 453, 380, 381, 365, 314,
                        90, 600, 594, 120, 333, 631, 301, 159, 477, 489, 560, 307, 4, 353,
                        386, 399, 8, 374, 638, 199, 454, 0, 578, 249, 418, 487, 564, 660, 535, 266, 28, 211, 559, 15,
                        591, 603, 313, 178, 536, 571, 441, 108, 361, 254, 277, 597, 526, 25, 335, 129, 217, 143, 45,
                        231, 116, 612, 206, 604, 195, 111, 232, 420, 79, 650, 369, 606,
                        438, 235, 259, 588, 281, 50, 527, 295, 537, 452, 542, 12, 375, 421, 572, 208, 634, 269, 350,
                        167, 628, 502, 26, 244, 540, 617, 342, 590, 1140, 892, 1030, 723, 1011, 876, 687, 912, 1274,
                        1112, 750, 1047, 1062, 1144, 950, 698, 921, 913, 1150, 919, 976, 761,
                        1231, 795, 1104, 1049, 764, 1260, 781, 1085, 1041, 1036, 1033, 1176, 1242, 1070, 724, 1097,
                        682, 1236, 721, 664, 989, 1074, 715, 1268, 819, 680, 1169, 1238, 1263, 738, 994, 1170, 1222,
                        1237, 896, 743, 1132, 1053, 807, 1120, 696, 946, 1232, 1068, 1188, 841,
                        1196, 673, 1095, 1023, 877, 884, 740, 813, 766, 796, 923, 1044, 741, 792, 987, 1189, 1130,
                        1057, 962, 669, 812, 1090, 1055, 1060, 862, 1008, 984, 775, 725, 1014, 929, 980, 940, 1227,
                        870, 955, 753, 1185, 712, 770, 1173, 1035, 883, 1077, 998, 991, 791,
                        1118, 855, 1184, 1099, 903, 956, 1087, 1398, 1315, 1282, 1343, 1354, 1342, 1369, 1390, 1430,
                        1284, 1352, 1393, 1400, 1445, 1327, 1335, 1304, 1295, 1448, 1427, 1421, 1288, 1452, 1416,
                        1338, 1339, 1404, 1291, 1329, 1434, 1447, 1319, 1277, 1449, 1323, 1276,
                        1528, 1467, 1460, 1551, 1552, 1464, 1462, 1535, 1554, 1559, 1588, 1530, 1520, 1507, 1582,
                        1477, 1569, 1543, 1482, 1531, 1544, 1459, 1476, 1512, 1526, 1498]
        testindex = sorted([t * 8 + i for t in testindex_interval for i in range(8)])

        if self.FLAG == 'TEST':
            self.pose = torch.from_numpy(np.concatenate([pose1[testindex], pose2[testindex]])).float()
            self.pose_label = np.concatenate([[1] * 132 * 8, [2] * 122 * 8, [3] * 36 * 8, [4] * 26 * 8, [0] * len(testindex)])
        else:
            '''
            sampling_ranges = [(0, 663, 66), (663, 1275, 61),  (1275, 1459, 18),  (1459, 1590, 13)]
            random samples some data from the whole data for validation
            '''
            validindex_interval = [87, 575, 599, 66, 484, 440, 324, 318, 228, 435, 185, 5, 158, 568, 394, 616, 614, 172, 83, 507, 412, 284, 410, 
                          186, 602, 77, 336, 88, 340, 23, 552, 425, 304, 126, 226, 523, 155, 432, 262, 270, 157, 411, 118, 601, 177, 462,
                            302, 456, 113, 181, 511, 184, 424, 52, 22, 377, 152, 423, 161, 49, 264, 188, 530, 236, 426, 229, 842, 1071, 
                            966, 1126, 782, 1051, 779, 1177, 829, 1113, 732, 1175, 922, 780, 1168, 1201, 1194, 834, 1179, 793, 822, 1080, 
                            1052, 759, 995, 927, 1013, 963, 699, 668, 1241, 1203, 1253, 788, 1020, 996, 772, 677, 836, 1207, 1076, 789, 971, 
                            746, 1133, 830, 1114, 756, 722, 949, 900, 777, 979, 1066, 858, 891, 801, 1078, 1101, 704, 1246, 1305, 1414, 1402, 
                            1387, 1330, 1366, 1345, 1320, 1285, 1450, 1332, 1302, 1341, 1347, 1326, 1407, 1298, 1287, 1518, 1589, 1532, 1493, 
                            1471, 1466, 1461, 1570, 1567, 1560, 1504, 1513, 1577]
            validindex = sorted([t * 8 + i for t in validindex_interval for i in range(8)])
            # Find the indices that are not in the subset
            non_subset_indices = np.setdiff1d(np.arange(pose1.shape[0]), validindex+testindex)
            self.pose_training = torch.from_numpy(np.concatenate([pose1[non_subset_indices], pose2[non_subset_indices]])).float()
            self.pose_valid = torch.from_numpy(np.concatenate([pose1[validindex], pose2[validindex]])).float()
            self.pose_training_label = np.concatenate([[1] * 465 * 8, [2] * 429 * 8, [3] * 130 * 8, [4] * 92 * 8, [0] * non_subset_indices.shape[0]])
            self.pose_valid_label = np.concatenate([[1] * 66 * 8, [2] * 61 * 8, [3] * 18 * 8, [4] * 13 * 8, [0] * len(validindex)])
            
    def get_texture(self, ):
        if not hasattr(self, "texture"):
            texturefiles = []
            for i in range(1, 9):
                for j in range(1, 11):
                    texturefiles.append(os.path.join(self.args.TEXTUREPath, f'{TEXTURE_Neworder[i]}',
                                                     f'{TEXTURE_Neworder[i]}_{str(j).zfill(3)}.png'))
            self.texture = torch.cat([torch.Tensor(
                np.array(Image.open(texturefiles[i]).convert("RGB").resize(
                    (self.args.uv_size, self.args.uv_size))) / 255.).unsqueeze(0) for i in range(len(texturefiles))],
                                     dim=0)
            self.texturekey = [TEXTURE_Neworder[i] for i in range(1, 9) for j in range(1, 11)]
            self.texture_testindex = None # !!!!!!!!!!!!!!!!!!! Not implemented
            validindex = [24, 11, 61,  1, 78, 42, 34, 56] #np.random.choice(range(0, 80), 8, replace=False) 
            # # Find the indices that are not in the subset
            self.texture_trainindex = np.setdiff1d(np.arange(80), validindex)
            self.texture_validindex = np.array(validindex)

    def get_rotation_angle(self):
        self.rot_model_4gt = axis_angle_to_matrix(torch.Tensor([np.radians(90), 0, 0])).unsqueeze(0)

    def __len__(self):
        return self.length

    def setup_weak_render(self, device, image_size, init_dist=2.2):
        # Get a batch of viewing angles.
        cameras, renderer = setup_weak_render(image_size, faces_per_pixel=1, device=device)
        return cameras, renderer

    def random_SMAL_pose_params(self, flag, data_batch_size, interval=8, exclude = False):
        if self.FLAG == 'TEST':
            selected_pose_set = self.pose 
            selected_pose_label = self.pose_label 
        elif self.FLAG == 'TRAIN':
            selected_pose_set = self.pose_training 
            selected_pose_label = self.pose_training_label 
        else:
            selected_pose_set = self.pose_valid 
            selected_pose_label = self.pose_valid_label 
        data_len = int(selected_pose_set.shape[0])
        index = np.random.randint(low=0, high=int(data_len / interval), size=(data_batch_size))
        pose_selected = selected_pose_set[::interval, :][index, :]
        pose_class = selected_pose_label[::interval][index]
        return pose_selected, pose_class

    def random_SMAL_shape_params(self, flag, data_batch_size):
        shape_selected = torch.tensor(np.random.normal(0, 1, size=(data_batch_size, 9))).float()
        shape_class = np.array([0  for i in range(data_batch_size)])
        return shape_selected, shape_class

    def random_texture(self, flag, data_batch_size):
        if self.FLAG == 'TEST':
            selected_texture_label = self.texture_testindex
            raise NotImplementedError
        elif self.FLAG == 'TRAIN':
            selected_texture_label = self.texture_trainindex
        else:
            selected_texture_label = self.texture_validindex
        index = np.random.randint(low=0, high=len(selected_texture_label), size=(data_batch_size))
        texture_class = selected_texture_label[index]
        init_texture_tensor =self.texture[texture_class]
        return init_texture_tensor, texture_class

    def random_background(self, flag, data_batch_size):
        if self.FLAG == 'TEST':
            selected_background_set = self.background_test
            selected_background_index = self.background_testdindex
        elif self.FLAG == 'TRAIN':
            selected_background_set = self.background_train
            selected_background_index = self.background_trainingindex
        else:
            selected_background_set = self.background_valid
            selected_background_index = self.background_validindex

        index = np.random.randint(low=0, high=int(len(selected_background_set)), size=(data_batch_size))
        background_name = [selected_background_set[i] for i in index]
        return background_name

    def obtain_SMAL_vertices(self, data_batch_size, poseflag, shapeflag, textureflag, interval=8, cameraindex=0):
        shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=data_batch_size)
        pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=data_batch_size,
                                                                interval=interval)
        texture_selected, texture_class = self.random_texture(flag=textureflag, data_batch_size=data_batch_size)
        pose_selected = self.get_pose_gt(pose_selected, cameraindex)
        verts, _, _ = self.smal_model(beta=shape_selected.to(self.device),
                                      theta=pose_selected.to(self.device),
                                      trans=torch.zeros((data_batch_size, 3)).to(self.device))
        trans = self.get_trans_gt(verts, shapeflag)
        verts = verts + trans
        return verts, texture_selected, shapeclass, poseclass, texture_class, shape_selected, pose_selected, trans

    def obtain_SMAL_meshes(self, data_batch_size, verts, texture_selected):
        assert verts.shape[0] == data_batch_size
        textures = TexturesUV(maps=texture_selected,
                              faces_uvs=self.ft.repeat(data_batch_size, 1, 1),
                              verts_uvs=self.vt.repeat(data_batch_size, 1, 1), sampling_mode="nearest").to(self.device)
        torch_mesh = Meshes(verts=verts, faces=self.faces.repeat(verts.shape[0], 1, 1), textures=textures).to(
            self.device)
        return torch_mesh

    def obtain_SMAL_pair_w_texture(self, label, data_batch_size, poseflag, shapeflag, textureflag, interval=8,cameraindex=0):
        if label == 1:  # label to 1: pose space: change pose ; only one cam one texture, but two SMAL (with same root)
            pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=data_batch_size,
                                                                    interval=interval) 
            pose_selected[1, :3] = pose_selected[0, :3]  # with the same root
            shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=1)
            shape_selected = shape_selected.repeat(data_batch_size, 1)
            shapeclass = shapeclass.repeat(data_batch_size)
            texture_selected, textureclass = self.random_texture(flag=textureflag, data_batch_size=1)
            texture_selected = texture_selected.repeat(data_batch_size, 1, 1, 1)
            textureclass = textureclass.repeat(data_batch_size)
        elif label == 2:  # label to 2: appearance space: change appearance; only one SMAL one cam, but two texture
            shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=data_batch_size)
            texture_selected, textureclass = self.random_texture(flag=textureflag, data_batch_size=data_batch_size)

            pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=1, interval=interval)
            pose_selected = pose_selected.repeat(data_batch_size, 1)
            poseclass = poseclass.repeat(data_batch_size)
        elif label == 3:  # label to 3: cam space: change cam ; only one SMAL one texture but two cam
            shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=1)
            shape_selected = shape_selected.repeat(data_batch_size, 1)
            shapeclass = shapeclass.repeat(data_batch_size)
            pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=1, interval=interval)
            pose_selected = pose_selected.repeat(data_batch_size, 1)
            poseclass = poseclass.repeat(data_batch_size)
            texture_selected, textureclass = self.random_texture(flag=textureflag, data_batch_size=1)
            texture_selected = texture_selected.repeat(data_batch_size, 1, 1, 1)
            textureclass = textureclass.repeat(data_batch_size)

        pose_selected = self.get_pose_gt(pose_selected, cameraindex)
        verts, _, _ = self.smal_model(beta=shape_selected.to(self.device),
                                      theta=pose_selected.to(self.device),
                                      trans=torch.zeros((data_batch_size, 3)).to(self.device))
        trans = self.get_trans_gt(verts, shapeflag)
        verts = verts + trans
        torch_mesh = self.obtain_SMAL_meshes(data_batch_size, verts, texture_selected)
        return torch_mesh, shapeclass, poseclass, textureclass, shape_selected, pose_selected, trans

    def get_image(self, data_batch_size, poseflag=None, shapeflag=None, textureflag=None, interval=8, cameraindex=None):
        verts, textures, shapeclass, poseclass, textureclass, shape_selected_gt, pose_selected_gt, trans_gt = [], [], [], [], [], [],[],[]
        for i in range(data_batch_size):
            if cameraindex is None:
                cameraindex = 1 
            v, texture, shapecla, posecla, texturecla, shape_selected, pose_selected, trans = self.obtain_SMAL_vertices(1,  poseflag = None, shapeflag = None, textureflag = None, interval = interval,
                                                                                  cameraindex=cameraindex)
            verts.append(v)
            textures.append(texture)
            shapeclass.append(shapecla)
            poseclass.append(posecla)
            textureclass.append(texturecla)
            shape_selected_gt.append(shape_selected)
            pose_selected_gt.append(pose_selected)
            trans_gt.append(trans)
        verts = torch.cat(verts)
        textures = torch.cat(textures)
        shapeclass = np.concatenate(shapeclass)
        poseclass = np.concatenate(poseclass)
        textureclass = np.concatenate(textureclass)
        shape_selected_gt = torch.cat(shape_selected_gt)
        pose_selected_gt = torch.cat(pose_selected_gt)
        trans_gt = torch.cat(trans_gt)
        meshes = self.obtain_SMAL_meshes(data_batch_size=data_batch_size, verts=verts, texture_selected=textures)
        return meshes, shapeclass, poseclass, textureclass, [[0] * data_batch_size][0], shape_selected_gt, pose_selected_gt, trans_gt

    def get_image_pair_label(self, data_batch_size, poseflag=None, shapeflag=None, textureflag=None, interval=8,
                             cameraindex=None):
        probablity = np.random.uniform(0, 1)
        # Determine the class
        if probablity < 1 / 3.:
            label = 1  # label to 1: pose space: change pose ;
        elif 1 / 3. <= probablity < 2 / 3.:
            label = 2  # label to 2: appearance space: change appearance
        else:
            label = 3  # label to 3: cam space: change cam
        if label == 1 or label == 2:
            cameraindex = 0 
        elif label == 3:
            cameraindex = 1 
    
        meshes, shapeclass, poseclass, textureclass, shape_selected, pose_selected, trans = self.obtain_SMAL_pair_w_texture(label, data_batch_size, poseflag = None,
                                                                                      shapeflag= None, textureflag = None,
                                                                                      interval=interval,
                                                                                      cameraindex=cameraindex)
        label = [[label] * data_batch_size][0]
        return meshes, shapeclass, poseclass, textureclass, label, shape_selected, pose_selected, trans

    def obtain_camera_label(self, kp_3d_tensor):
        batch = kp_3d_tensor.shape[0]  # [N, 17, 3]
        cameraclass = []
        for i in range(batch):
            kp_3d_now = kp_3d_tensor[i, ...]
            if kp_3d_now[2, 0] < kp_3d_now[4, 0]:  # nose x < tail x
                left = 1;
                right = 0
            else:
                right = 1;
                left = 0
            if kp_3d_now[2, 2] > kp_3d_now[4, 2]:  # nose z> tail z
                away = 1;
                toward = 0
            else:
                toward = 1;
                away = 0
            cameraclass.append([toward, away, left, right])
        return np.array(cameraclass)

    def get_pose_gt(self, pose_selected, cameraindex):
        # rotate the model given the render axis and camera angle
        # rotate the model to y axis is height
        pose_origianl = pose_selected[:, :3]
        pose_origianl_matrix = axis_angle_to_matrix(pose_origianl) 
        
        if cameraindex == 0:
            # only sample one camera angle
            angle_random = (np.random.random(size = 1) *360).repeat(pose_selected.shape[0])
        else:
            # sample pose_selected.shape[0] camera angles
            angle_random = np.random.random(size = pose_selected.shape[0]) *360
        rot_candidate_4gt = axis_angle_to_matrix(torch.Tensor([[0, -np.radians(i), 0] for i in angle_random]))
        pose_updated_matrix = torch.matmul(torch.matmul(rot_candidate_4gt, self.rot_model_4gt),pose_origianl_matrix)  # [1,3,3] #
        pose_update = rotmat_to_axis_angle(pose_updated_matrix.unsqueeze(1),number=1)  # current the same results as in pytorch3D: pose_update = matrix_to_axis_angle(pose_updated_matrix)
        pose_selected[:, :3] = pose_update
        #### pose update done #####################################
        return pose_selected

    def get_trans_gt(self, verts, shapeflag):
        trans = torch.mean(verts, axis=1) 
        trans = -trans + torch.tensor([[0., 0., 65]]).float().to(self.device)
        return trans.unsqueeze(1)

    def change_background(self,init_images_tensor, mask_image_tensor, background_name):
        '''
        mask_image_tensor [B,1,256,256]; init_images_tensor[..., :3] [B,3,256,256]; background_name: [B]
        '''
        new_images = torch.zeros_like(init_images_tensor)
        # Iterate over each image in the batch
        for i in range(init_images_tensor.shape[0]):
            # Read the background image
            background_PIL = Image.open( background_name[i]).convert("RGB")
            background_tensor = self.background_transform(background_PIL)
            mask_expanded = mask_image_tensor[i].expand_as(init_images_tensor[i])
            # Use the mask to select foreground
            foreground = init_images_tensor[i] * mask_expanded
            # Use the inverted mask to select background
            background = background_tensor * (1 - mask_expanded)
            # Combine
            new_images[i] = foreground + background        
        return new_images

    def get_real_image(self, data_batch_size):
        assert self.FLAG in ['TRAIN'] # only used for training
        if self.FLAG == 'TRAIN':
            selected_realimg_filelist = self.realimg_filelist_train
        else:
            selected_realimg_filelist = self.realimg_filelist_eval
        selected_index = np.random.randint(low=0, high=len(selected_realimg_filelist), size=(data_batch_size))
        img_list, mask_list, joint_list, labels_list = [],[],[],[]
        for index in selected_index:
            if self.args.REALDATASET == 'Animal3D':
                img, mask, joints, labels = self.get_real_image_animal3d(index) 
            elif self.args.REALDATASET == 'MagicPony':
                img, mask, joints, labels = self.get_real_image_magicpony(index) 
            img_list.append(img)
            mask_list.append(mask)
            joint_list.append(joints) # [17,3] 
            labels_list.append(labels)
        img_list = np.stack(img_list, axis=0) # [B,256,256, 3]
        mask_list = np.stack(mask_list, axis=0) # [B,256,256,1]
        joint_list = np.stack(joint_list, axis=0) # [B,17,3]
        labels_list = np.array(labels_list) # [B,17,3]

        temp_images_tensor = torch.tensor(img_list).permute(0,3,1,2).float() #[B,3,256,256]
        mask_image_tensor = torch.tensor(mask_list).permute(0,3,1,2).float() #[B,1,256,256]
        kp_2d_tensor = torch.tensor(joint_list).float() #[B,17,3]
        kp_3d_tensor = torch.zeros((data_batch_size, 17, 4)).float() #[B,17,3]
        shape_gt_tensor = torch.zeros((data_batch_size, 9)).float() #[B,9]
        pose_gt_tensor = torch.zeros((data_batch_size, 108)).float() #[B,108]
        trans_gt_tensor = torch.zeros((data_batch_size, 1, 3)).float() #[B,1,3]

        shapeclass = np.array([-1 for i in range(data_batch_size)])
        poseclass = np.array([-1 for i in range(data_batch_size)])
        textureclass = np.array([-1 for i in range(data_batch_size)])
        return temp_images_tensor, mask_image_tensor, kp_2d_tensor,shapeclass,poseclass,textureclass, \
            kp_3d_tensor,labels_list, shape_gt_tensor, pose_gt_tensor, trans_gt_tensor

    def get_real_image_animal3d(self, index):
        assert self.FLAG in ['TRAIN'] # only used for training
        if self.FLAG == 'TRAIN':
            selected_realimg_filelist = self.animal3drealimg_filelist_train
            selected_realimg_anno = self.animal3drealanno_dict_train
        else:
            selected_realimg_filelist = self.animal3drealimg_filelist_eval
            selected_realimg_anno = self.animal3drealanno_dict_eval  
        # read image
        img_id   = selected_realimg_filelist[index]
        img_path = selected_realimg_anno[str(img_id)]['image_path']
        img = imageio.v2.imread(img_path) / 255.0 #[256,256,3]
        if len(img.shape) == 2: # for grayscale images
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        joints = np.array(selected_realimg_anno[img_id]['kp']) #[16,3]
        # get mask (optionally)
        if 'mask' not in selected_realimg_anno[img_id].keys():
                mask = np.zeros((256, 256,1))
        else:
            mask = np.expand_dims(selected_realimg_anno[img_id]['mask'], 2) #[256,256,1]
        if self.FLAG == 'TRAIN':
            if np.random.rand(1) > 0.5: # mirror the image
                # Need copy bc torch collate doesnt like neg strides
                img = img[:, ::-1, :].copy() #[256,256, 3]
                mask = mask[:, ::-1].copy() #[256,256]
                # Flip joints
                new_x = img.shape[1] - joints[:, 0] - 1
                joints_flip = np.hstack((new_x[:, None], joints[:, 1:]))
                joints_flip = joints_flip[QUAD_JOINT_PERM, :]
                joints = joints_flip
        joints = np.vstack([joints, np.zeros((1, 3))])
        label = 5
        return img, mask, joints, label
        
    def get_real_image_magicpony(self, index):
        assert self.FLAG in ['TRAIN'] # only used for training
        if self.FLAG == 'TRAIN':
            selected_realimg_filelist = self.magicponytrainfilelist
        else:
            selected_realimg_filelist = self.magicponyvalidfilelist

        # read image
        img_path   = selected_realimg_filelist[index]
        img = imageio.v2.imread(img_path) / 255.0 #[256,256,3]
        mask_path = img_path.replace('rgb', 'mask').replace('.jpg', '.png')
        joint_path = img_path.replace('rgb', 'kp').replace('.jpg', '.npy')

        mask = imageio.v2.imread(mask_path) / 255.0 
        mask = mask[:,:,[0]]#[256,256,1]

        joints = np.load(joint_path, allow_pickle=True).item()['keypoints'].astype(np.float32) #[17,3]
        if np.random.rand(1) > 0.5: # mirror the image
            # Need copy bc torch collate doesnt like neg strides
            img = img[:, ::-1, :].copy() #[256,256, 3]
            mask = mask[:, ::-1].copy() #[256,256]
            # Flip joints
            new_x = img.shape[1] - joints[:, 0] - 1
            joints_flip = np.hstack((new_x[:, None], joints[:, 1:]))
            joints_flip = joints_flip[JOINT_PERM, :]
            joints = joints_flip
        label = 4
        return img, mask, joints, label

    def __getitem__(self, idx):
        # Generate data using the provided function
        '''
        init_images_tensor: torch.Size([36, 256, 256, 4])
        mask_image_tensor: torch.Size([36, 256, 256, 1])
        kp_2d_tensor: torch.Size([36, 17, 3])

        out:
        init_images_tensor: torch.Size([1, 36, 3, 256, 256])
        mask_image_tensor: torch.Size([1, 36, 1, 256, 256])
        kp_2d_tensor: torch.Size([1, 36, 17, 3])
        '''
        if not hasattr(self, "renderer") or not hasattr(self, "cameras"):
            self.cameras, self.renderer = self.setup_weak_render(device=self.device,
                                                                 image_size=self.args.imgsize)
        
        probablity = np.random.uniform(0, 1)
        # probablity = 0
        if probablity < 0.5 and self.FLAG == 'TRAIN': # get real image only during training
            temp_images_tensor, mask_image_tensor, kp_2d_tensor,shapeclass,poseclass,textureclass, \
            kp_3d_tensor,label, shape_selected_gt,pose_selected_gt,trans_gt = self.get_real_image(self.args.data_batch_size) 
        else:
            if self.args.getPairs:
                meshes, shapeclass, poseclass, textureclass, label, shape_selected_gt, pose_selected_gt, trans_gt = self.get_image_pair_label(
                    data_batch_size=self.args.data_batch_size,
                    poseflag=None, shapeflag=None, textureflag=None,
                    interval=self.args.useinterval)
            else:
                meshes, shapeclass, poseclass, textureclass, label, shape_selected_gt, pose_selected_gt, trans_gt = self.get_image(self.args.data_batch_size,
                                                                                    poseflag=None, shapeflag=None,
                                                                                    textureflag=None,
                                                                                    interval=self.args.useinterval,
                                                                                    cameraindex=self.cameraindex)  # self.get_image_samebatch_with_same_flag

            init_images_tensor, mask_image_tensor, kp_3d_tensor, kp_2d_tensor = render_image_mask_and_save_with_preset_render(
                renderer=self.renderer,
                cameras=self.cameras,
                mesh=meshes,
                image_size=self.args.imgsize,
                save_intermediate=False)
            init_images_tensor = init_images_tensor.detach().cpu()
            mask_image_tensor = mask_image_tensor.detach().cpu()
            kp_2d_tensor = kp_2d_tensor.detach().cpu()
            kp_3d_tensor = kp_3d_tensor.detach().cpu()

            shape_selected_gt = shape_selected_gt.detach().cpu()
            pose_selected_gt = pose_selected_gt.detach().cpu()
            trans_gt = trans_gt.detach().cpu()

            mask_image_tensor = mask_image_tensor.permute(0, 3, 1, 2)  
            temp_images_tensor = init_images_tensor[..., :3].permute(0, 3, 1, 2)  

            if self.args.background:
                background_name = self.random_background(flag = None, data_batch_size = self.args.data_batch_size)
                temp_images_tensor = self.change_background(temp_images_tensor, mask_image_tensor, background_name)

            del meshes

        if self.transform:
            init_images_tensor = self.process_image(temp_images_tensor)  

        shapeclass = torch.tensor(shapeclass.tolist(), dtype=int)
        poseclass = torch.tensor(poseclass.tolist(), dtype=int)
        textureclass = torch.tensor(textureclass.tolist(), dtype=int)
        camera_index_class = torch.tensor(
            self.obtain_camera_label(kp_3d_tensor))  
        label_tensor = torch.tensor(label, dtype=int)  

        out = (*map(none_to_nan, (
            init_images_tensor, mask_image_tensor, temp_images_tensor, shapeclass, poseclass, textureclass,
            camera_index_class, kp_3d_tensor, None, label_tensor, idx, kp_2d_tensor, shape_selected_gt,pose_selected_gt,trans_gt)),)  # for batch collation

        
        # Clear up the memory
        torch.cuda.empty_cache()
        # Call the garbage collector
        gc.collect()
        return out

if __name__ == '__main__':
    import torch.multiprocessing as mp

    # Set the start method to 'spawn'
    mp.set_start_method('spawn', force=True)
    import random, os, argparse, time
    import numpy as np
    from src.utils.misc import validate_tensor_to_device, validate_tensor, collapseBF
    from PIL import Image, ImageDraw


    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_dir', type=str,
                            default='/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models',
                            help='model dir')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--imgsize', type=int, default=256, help='image size')
        parser.add_argument('--seed', type=int, default=0, help='max. number of training epochs')

        parser.add_argument('--PosePath', type=str, default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/pose',
                            help='model dir')
        parser.add_argument("--TEXTUREPath", type=str,
                            default="/home/x_cili/x_cili_lic/DESSIE/data/syndata/TEXTure")
        parser.add_argument('--uv_size', type=int, default=256, help='uv size')
        parser.add_argument('--data_batch_size', type=int, default=2, help='batch size')

        parser.add_argument('--useinterval', type=int, default=8, help='number of interval of the data')
        parser.add_argument("--getPairs", action="store_true", default=False,
                            help="get image pair with label")
        
        parser.add_argument("--background", action="store_true", default=False,
                           help="get image pair with label")
        parser.add_argument("--background_path", default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/coco', help="")
        
        parser.add_argument("--REALDATASET", default='MagicPony', help="Animal3D (Staths) or MagicPony")
        parser.add_argument("--REALPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/realimg', help="path to Staths Dataset")
        parser.add_argument("--web_images_num", type=int, default=0, help="Staths dataset")
        parser.add_argument("--REALMagicPonyPATH", default='/home/x_cili/x_cili_lic/CrossSpecies/data/magicpony', help="magicpony dataset")
        args = parser.parse_args()
        return args


    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.background = True
    args.getPairs = True # getPairs is true then data_batch_size is 2
    args.batch_size = 4
    args.data_batch_size = 2
    print(args)
    dataset = DessiePIPEWithRealImage(args=args, device=device, length=1000, FLAG='TRAIN')
    
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True)    
    VISUAL = True
    save_path = '/home/x_cili/x_cili_lic/DESSIE/results/dataloader_realimg'
    start = time.time()
    for i, batch in enumerate(train_dl):
        print(i)
        input_image_tensor, mask_gt_tensor, denormalized_image_tensor, shapeclass_tensor, poseclass_tensor, textureclass_tensor, cameraclass_tensor, \
        kp_3d_tensor,_, label_tensor, frame_idx, kp2d_tensor, shape_gt_tensor, pose_gt_tensor, trans_gt_tensor = (*map(lambda x: validate_tensor_to_device(x, device), batch),)
        print(label_tensor.shape) #[batch_size,data_batch_size]
        # input_image_tensor, mask_gt_tensor, _, _, _, _, _, _, _, label_tensor, frame_idx, kp2d_tensor = (*map(lambda x: validate_tensor(x,), batch),)
        print(input_image_tensor.shape)  # [batch_size, data_batch_size, 3,256,256]
        print(shapeclass_tensor.shape)  # [batch_size, data_batch_size]
        print(label_tensor.shape)  # [batch_size, 1, data_batch_size]
        print(cameraclass_tensor.shape)  # [batch_size, data_batch_size, 4]
        print(shape_gt_tensor.shape) #[batch, data_batch_size, 9]
        print(pose_gt_tensor.shape) #[batch, data_batch_size, 108]
        print(trans_gt_tensor.shape) #[batch, data_batch_size, 1,3]
        input_image_tensor = collapseBF(input_image_tensor)
        mask_gt_tensor = collapseBF(mask_gt_tensor)
        kp2d_tensor = collapseBF(kp2d_tensor)
        denormalized_image_tensor = collapseBF(denormalized_image_tensor)
        shapeclass = collapseBF(shapeclass_tensor)
        poseclass = collapseBF(poseclass_tensor)
        textureclass = collapseBF(textureclass_tensor)
        cameraclass = collapseBF(cameraclass_tensor).cpu().data.numpy().tolist()
        print(collapseBF(cameraclass_tensor).shape)  # [batch_size *data_batch_size, 4]
        label_tensor_collapse = collapseBF(label_tensor)
        print(label_tensor_collapse.shape)  # [batch_size *data_batch_size]
        if VISUAL:
            if i > 4:
                break
            batch_size = denormalized_image_tensor.shape[0]
            denormalized_image = denormalized_image_tensor.cpu()  # [B, 3, I, I]
            init_image = [transforms.ToPILImage()(denormalized_image[t]).convert("RGB") for t in range(batch_size)]
            kp2d = kp2d_tensor.cpu()
            label = label_tensor.cpu()
            # Draw each keypoint as a circle
            for t in range(batch_size):
                # Create a drawing context
                draw = ImageDraw.Draw(init_image[t])
                for tt, keypoint in enumerate(kp2d[t]):
                    # For a circle, we need the top-left and bottom-right coordinates of the bounding square
                    x, y, flag = keypoint
                    if flag == 1 or flag >=0.5:
                        r = 2  # radius of the circle
                        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
                        # drawing text size
                # if args.getPairs:
                    # draw.text((5, 5), text=f"label:{label[t, 0]}", fill=(255,0,0), font=ImageFont.load_default(), align="left")
                draw.text((5, 5), text=f"label:{label_tensor_collapse[t]}", fill=(255,0,0), font=ImageFont.load_default(), align="left")

            mask_image = mask_gt_tensor[:, 0, ...].cpu()  # [B, I, I, 3]
            mask_image = [transforms.ToPILImage()(mask_image[t]).convert("L") for t in range(batch_size)]

            if args.getPairs:
                for t in range(args.batch_size):
                    concatenated_img = Image.new('RGB', (init_image[t].width * 4, init_image[t].height))
                    concatenated_img.paste(init_image[t * 2], (0, 0))
                    concatenated_img.paste(mask_image[t * 2], (init_image[t].width, 0))
                    concatenated_img.paste(init_image[t * 2 + 1], (init_image[t].width * 2, 0))
                    concatenated_img.paste(mask_image[t * 2 + 1], (init_image[t].width * 3, 0))
                    concatenated_img.save(os.path.join(save_path, f"{str(i).zfill(3)}_{str(t).zfill(3)}_image.png"))
            else:
                for t in range(batch_size):
                    concatenated_img = Image.new('RGB', (init_image[t].width * 2, init_image[t].height))
                    concatenated_img.paste(init_image[t], (0, 0))
                    concatenated_img.paste(mask_image[t], (init_image[t].width, 0))
                    concatenated_img.save(os.path.join(save_path, f"{str(i).zfill(3)}_{str(t).zfill(3)}_image.png"))
        print(time.time() - start)