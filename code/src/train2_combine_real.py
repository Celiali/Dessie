import sys, os
sys.path.append(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging
)
from src.trainer.vanillaupdatetrainer_realimg import VanillaUpdateRealImageTrainer
from src.dataset2.basedataset8 import DessiePIPEWithRealImage
import os, random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_file', default= None, required=False, help='checkpoint for resuming')

    parser.add_argument('--train', type=str, required=False, help='Train or Test')

    parser.add_argument('--data_dir', type=str, default='',help='')
    parser.add_argument('--model_dir', type=str,
                        default='/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models', help='model dir')

    parser.add_argument('--save_dir', type=str, default='/home/x_cili/x_cili_lic/DESSIE/results/model',help='save dir')
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')
    parser.add_argument('--ModelName', type=str, help='model name')
    parser.add_argument('--DatasetName', type=str, default='None', help='')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--imgsize', type=int, default=256, help='number of workers')

    parser.add_argument('--W_shape_prior', default=50., type=float, help='shape prior')
    parser.add_argument('--W_kp_img', default=0.001, type=float, help='kp loss for image')
    parser.add_argument('--W_mask_img', default=0.0001, type=float, help='mask loss for image: 0.0001 or 1')
    parser.add_argument('--W_pose_img', default=0.01, type=float, help='pose prior for image')
    
    parser.add_argument('--W_l2_shape_1', default=0., type=float, help='Dloss latent label1')
    parser.add_argument('--W_l2_pose_2', default=0., type=float, help='Dloss latent label2')
    parser.add_argument('--W_l2_shape_3', default=0., type=float, help='Dloss latent label3')
    parser.add_argument('--W_l2_pose_3', default=0., type=float, help='Dloss latent label3')
    parser.add_argument('--W_l2_rootrot_1', default=0, type=float, help='Dloss value label1')
    parser.add_argument('--W_l2_rootrot_2', default=0, type=float, help='Dloss value label2')
    
    parser.add_argument('--lr', type=float, default=5e-05, help='optimizer learning rate')
    parser.add_argument('--max-epochs', type=int, default=1000, help='max. number of training epochs')

    parser.add_argument('--seed', type=int, default=0, help='max. number of training epochs')

    parser.add_argument('--PosePath', type=str, default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/pose')
    parser.add_argument("--TEXTUREPath", type=str, default="/home/x_cili/x_cili_lic/DESSIE/data/syndata/TEXTure")
    parser.add_argument('--uv_size', type=int, default=256, help='number of workers')
    parser.add_argument('--data_batch_size', type=int, default=2, help='batch size; before is 36')

    parser.add_argument('--useSynData', action="store_true", help="True: use syndataset")
    parser.add_argument('--useinterval', type=int, default=8, help='number of interval of the data')
    parser.add_argument("--getPairs", action="store_true", default=False,help="get image pair with label")
    
    parser.add_argument("--TEXT", action="store_true", default=False,help="Text label input")
    
    # For DINO
    parser.add_argument("--DINO_model_name", type=str, default="dino_vits8")
    parser.add_argument("--DINO_frozen", action="store_true", default=False,help="frozen DINO")
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False,help="obtain CLS token or use key")

    # For GT
    parser.add_argument("--GT", action="store_true", default=False,help="obtain gt or not")
    parser.add_argument('--W_gt_shape', default=0, type=float, help='weight for gt')
    parser.add_argument('--W_gt_pose', default=0., type=float, help='weight for gt')
    parser.add_argument('--W_gt_trans', default=0., type=float, help='weight for gt')

    parser.add_argument("--pred_trans", action="store_true", default=False,help="model to predict translation or not")

    parser.add_argument("--background", action="store_true", default=False,help="get image pair with label")
    parser.add_argument("--background_path", default='/home/x_cili/x_cili_lic/DESSIE/data/coco', help="")

    parser.add_argument("--REALDATASET", default='MagicPony', help="Animal3D or MagicPony")
    parser.add_argument("--REALPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/realimg', help="staths dataset")
    parser.add_argument("--web_images_num", type=int, default=0, help="staths dataset")
    parser.add_argument("--REALMagicPonyPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/magicpony', help="magicpony dataset")
    args = parser.parse_args()
    return args

PREDEFINED_SEED = 0

def worker_init_fn(worker_id):
    # Set the worker seed to `torch.initial_seed() + worker_id`
    worker_seed = PREDEFINED_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_syn_dataset(args, device, train_length=1000, valid_length=30):
    if args.DatasetName == 'DessiePIPEWithRealImage':
        dataset = DessiePIPEWithRealImage(args=args, device=device, length=train_length, FLAG = 'TRAIN')
        validdataset = DessiePIPEWithRealImage(args=args, device=device, length=valid_length, FLAG = 'VALID')
    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=True,pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(validdataset, batch_size=args.batch_size, num_workers=0, shuffle=True,pin_memory=True, drop_last=True)
    return train_loader, val_loader

def main(args):
    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    
    if args.useSynData:
        if args.getPairs:
            # using imagePair
            train_length = 3200
            valid_length = 320
        else:
            raise NotImplementedError   
        train_loader, val_loader = get_syn_dataset(args, device, train_length, valid_length)
    else:
        raise NotImplementedError

    # initialize model
    if args.ckpt_file == None:
        vanilla = VanillaUpdateRealImageTrainer(args)
    else:
        vanilla = VanillaUpdateRealImageTrainer(args)
        checkpoint = torch.load(args.ckpt_file, map_location=torch.device('cpu'))
        vanilla.load_state_dict(checkpoint['state_dict'])
        print('Loaded model from checkpoint:', args.ckpt_file)

    # create trainer
    logger = TensorBoardLogger(args.save_dir, name=args.name, version=args.version)
    logger.log_hyperparams(vars(args))  # save all (hyper)params

    ckpt_callback = ModelCheckpoint(
        filename='best',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    callbacks_list = [ckpt_callback]
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks_list,
        accelerator=accelerator,
        devices=num_gpus,  
        max_epochs=args.max_epochs,
        gradient_clip_val=None,  
        gradient_clip_algorithm=None,  
        log_every_n_steps=int(len(train_loader) / num_gpus),
        enable_progress_bar=True,
        strategy='ddp' if num_gpus > 1 else None,
    )

    # train model
    trainer.fit(
        vanilla,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=None
    )


if __name__ == '__main__':
    ##### syn dataset
    import torch.multiprocessing as mp
    ##https://github.com/Lightning-AI/pytorch-lightning/issues/17026
    # Set the start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    args = parse_args()
    main(args)