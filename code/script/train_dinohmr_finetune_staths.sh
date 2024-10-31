#!/bin/bash
#SBATCH --gpus 2
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mem 50G
PROJECTPATH=/home/x_cili/x_cili_lic/DESSIE
cd $PROJECTPATH/code/src

srun /home/x_cili/.conda/envs/lassie/bin/python train2_combine_real.py --W_shape_prior 0.01 --W_kp_img 0.001 --W_pose_img 0.01 --W_mask_img 0.0001 \
                                                        --name TEST \
                                                        --save_dir $PROJECTPATH/results/model \
                                                        --model_dir $PROJECTPATH/code/src/SMAL/smpl_models \
                                                        --PosePath $PROJECTPATH/data/syndata/pose \
                                                        --TEXTUREPath $PROJECTPATH/data/syndata/TEXTure \
                                                        --useSynData --useinterval 8 --max-epochs 400 --TEXT \
                                                        --DINO_model_name dino_vits8 --imgsize 256 \
                                                        --GT --W_gt_shape 0. --W_gt_pose 0. --W_gt_trans 0. \
                                                        --ModelName DINOHMR --pred_trans \
                                                        --background --background_path $PROJECTPATH/data/syndata/coco \
                                                        --getPairs --batch_size 16 --data_batch_size 2 \
                                                        --W_l2_shape_1 0. --W_l2_pose_2 0. --W_l2_shape_3 0. --W_l2_pose_3 0. \
                                                        --W_l2_rootrot_1 0.0 --W_l2_rootrot_2 0.0 \
                                                        --DatasetName DessiePIPEWithRealImage \
                                                        --ckpt_file $PROJECTPATH/results/model/TOTALRANDOM/version_12/checkpoints/best.ckpt \
                                                        --REALDATASET Animal3D --web_images_num 0 
