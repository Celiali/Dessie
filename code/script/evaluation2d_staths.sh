#!/bin/bash
PROJECTPATH=/home/x_cili/x_cili_lic/DESSIE
cd $PROJECTPATH/code/src

VALUE=1

DATA=$PROJECTPATH/data/staths
FILE=evaluate_2d_staths.py

######## DinoHMR
MODELNAME=DINOHMR
EXPNAME=TOTALRANDOM
CKPT=$PROJECTPATH/results/model/$EXPNAME/version_12/checkpoints/best.ckpt
######## DinoHMR*  
# MODELNAME=DINOHMR
# EXPNAME=COMBINAREAL
# CKPT=$PROJECTPATH/results/model/$EXPNAME/version_9/checkpoints/best.ckpt

echo $CKPT
/home/x_cili/.conda/envs/lassie/bin/python $FILE \
                                                --name $EXPNAME --batch_size $VALUE --data_batch_size 1 \
                                                --useSynData --TEXT --getPairs \
                                                --ModelName $MODELNAME --pred_trans \
                                                --ckpt_file $CKPT --data_dir $DATA 

######## Dessie
MODELNAME=DESSIE
EXPNAME=TOTALRANDOM
CKPT=$PROJECTPATH/results/model/$EXPNAME/version_9/checkpoints/best.ckpt
######## Dessie*  
# MODELNAME=DESSIE
# EXPNAME=COMBINAREAL
# CKPT=$PROJECTPATH/results/model/$EXPNAME/version_8/checkpoints/best.ckpt

echo $CKPT
/home/x_cili/.conda/envs/lassie/bin/python $FILE  \
                                                --name $EXPNAME --batch_size $VALUE --data_batch_size 1 \
                                                --useSynData --TEXT --getPairs \
                                                --ModelName $MODELNAME --pred_trans \
                                                --ckpt_file $CKPT --data_dir $DATA 