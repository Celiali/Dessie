'''
visualize dino keys
'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
import torch
import os, random, math
import numpy as np
import cv2
import numpy as np
from PIL import Image
from src.train2 import parse_args
import glob
from sklearn.decomposition import PCA
from torchvision import transforms  as T 
from src.test import get_model

def use_key(extractor, img, denormalized_img, name, flag, save_path):

        # calculate the keys
    with torch.no_grad():
        keys = extractor.get_keys_from_input(img, 11)[0, :, 0:, :]  
    # reshape the reduced keys to the image shape
    patch_size = extractor.get_patch_size()
    patch_h_num = extractor.get_height_patch_num(img.shape)
    patch_w_num = extractor.get_width_patch_num(img.shape)
    num_head = extractor.get_head_num()
    # Determine total width and height
    # Create a new blank image
    total_width = patch_w_num * (num_head+1) * patch_size
    max_height = patch_h_num * patch_size
    concatenated_img = Image.new('RGB', (total_width, max_height))
    concatenated_img_mask = Image.new('RGB', (total_width, max_height))
    img_PIL = T.ToPILImage()(denormalized_img[0,...]).convert("RGB")
    concatenated_img.paste(img_PIL, (0, 0))
    # concatenated_img_mask.paste(img_PIL, (0, 0))
    key_image = []
    n_componets = 3
    for i in range(num_head):
        pca = PCA(n_components=n_componets)
        pca.fit(keys[i].transpose(1, 0).cpu().numpy())
        components = pca.components_[None, ...]
        if n_componets ==1 :
            pca_image = components[:, :, 1:].reshape(1, patch_h_num, patch_w_num).transpose(1, 2, 0)
        else:
            pca_image = components[:, :, 1:].reshape(n_componets, patch_h_num, patch_w_num).transpose(1, 2, 0)
        pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
        h, w, _ = pca_image.shape
        if n_componets ==1 :
            pca_image = Image.fromarray(np.uint8(pca_image * 255).repeat(3, axis = 2))  
        else:
            pca_image = Image.fromarray(np.uint8(pca_image * 255))
        pca_image = T.Resize((h * patch_size, w * patch_size), interpolation=T.InterpolationMode.BILINEAR)(pca_image)
        concatenated_img.paste(pca_image, ((i+1) * (w * patch_size), 0))
        
        pca_image_numpy = np.array(pca_image)
        key_image.append(pca_image_numpy)
    concatenated_img.save(f'{save_path}/{name}_key_{flag}.png')
        
def use_sim_key(extractor, img,name, flag, save_path):
    # calculate the keys
    with torch.no_grad():
        keys_self_sim = extractor.get_keys_self_sim_from_input(img, 11)
    n_componets = 3
    pca = PCA(n_components=n_componets)
    keys_self_sim_cpu = keys_self_sim[0].cpu().numpy()
    pca.fit(keys_self_sim_cpu)
    reduced = pca.transform(keys_self_sim_cpu)[None, ...]

    # reshape the reduced keys to the image shape
    patch_size = extractor.get_patch_size()
    patch_h_num = extractor.get_height_patch_num(img.shape)
    patch_w_num = extractor.get_width_patch_num(img.shape)
    pca_image = reduced[:, 1:].reshape(patch_h_num, patch_w_num, n_componets)
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
    h, w, _ = pca_image.shape
    if n_componets == 1:
        pca_image = Image.fromarray(np.uint8(pca_image.repeat(3, axis = 2) * 255))
    else:
        pca_image = Image.fromarray(np.uint8(pca_image * 255))
    pca_image = T.Resize((h * patch_size, w * patch_size), interpolation=T.InterpolationMode.BILINEAR)(pca_image)
    pca_image.save(f'{save_path}/{name}_simkey_{flag}.png')
    return pca_image


# define main function
if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    project_path = '/home/x_cili/x_cili_lic/DESSIE'
    # get data
    ###################### 
    image_path = sorted(glob.glob(os.path.join(project_path, 'data/demo_key/*.png')))
    print(image_path)
    save_path = os.path.join(project_path, 'results/key')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # USE_MODEL = 'original' 
    USE_MODEL = 'DESSIE'   
    if USE_MODEL == 'original':
        # original model
        from src.model.extractor import VitExtractor
        encoder = VitExtractor(model_name='dino_vits8', frozen = True).to(device)
        encoder.eval()
    elif USE_MODEL == 'DESSIE':            
        # get model
        args.ModelName='DESSIE' 
        args.pred_trans = True
        args.TEXT = True
        ckpt_file = os.path.join(project_path, 'results/model/TOTALRANDOM/version_9/checkpoints/best.ckpt' )
        Model, trainer = get_model(args, ckpt_file, device)
        Model.eval()
        encoder = Model.model.encoder

    with torch.no_grad():
        for i in range(0, len(image_path)):
            image = cv2.imread(image_path[i])/255.
            img = image[:, :, ::-1]
            w,h = img.shape[1], img.shape[0]    
            if w != 256 or h != 256:
                img = cv2.resize(img, (256, 256))
            denormalize_images = torch.from_numpy(img.copy()).permute(2,0,1)[None,:,:,:].float()
            input_image_tensor = T.Normalize(np.array([0.485, 0.456, 0.406]),np.array([0.229, 0.224, 0.225]))(denormalize_images).to(device)            
            # input image tensor [1, 3, 256, 256]
            # denormalize_images [1, 3, 256, 256]
            use_key(encoder, input_image_tensor, denormalize_images,os.path.basename(image_path[i]).split('.')[0], 
                    flag = USE_MODEL, save_path=save_path)
            use_sim_key(encoder, input_image_tensor,os.path.basename(image_path[i]).split('.')[0], 
                        flag = USE_MODEL, save_path=save_path)            