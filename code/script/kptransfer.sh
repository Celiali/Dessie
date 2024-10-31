#!/bin/bash
PROJECTPATH=/home/x_cili/x_cili_lic/DESSIE
cd $PROJECTPATH/code/src
### remember to add REALPATH
DATA=$PROJECTPATH/data/realimg/pascal_val
FILE=evaluate_kptransfer.py

# DinoHMR*
MODELNAME=DINOHMR
EXPNAME=COMBINAREAL
for t in 9 ; do 
    CKPT=$PROJECTPATH/results/model/$EXPNAME/version_$t/checkpoints/best.ckpt
    echo $CKPT
    /home/x_cili/.conda/envs/lassie/bin/python $FILE \
                                                    --name $EXPNAME --batch_size 1 --data_batch_size 1 \
                                                    --useSynData --TEXT --getPairs \
                                                    --ModelName $MODELNAME --pred_trans \
                                                    --ckpt_file $CKPT --data_dir $DATA 
                                                    
done

## Dessie*
MODELNAME=DESSIE
EXPNAME=COMBINAREAL
for t in 8 ; do
    CKPT=$PROJECTPATH/results/model/$EXPNAME/version_$t/checkpoints/best.ckpt
    /home/x_cili/.conda/envs/lassie/bin/python $FILE  \
                                                    --name $EXPNAME --batch_size 1 --data_batch_size 1 \
                                                    --useSynData --TEXT --getPairs \
                                                    --ModelName $MODELNAME --pred_trans \
                                                    --ckpt_file $CKPT --data_dir $DATA 
done