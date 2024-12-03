#!/bin/bash
PROJECTPATH=/home/x_cili/x_cili_lic/DESSIE
cd $PROJECTPATH/code/src

DATA=$PROJECTPATH/data/magicpony
FILE=evaluate_2d_magicpony.py

## DinoHMR
MODELNAME=DINOHMR
EXPNAME=TOTALRANDOM
for t in 12 ; do 
    CKPT=$PROJECTPATH/results/model/$EXPNAME/version_$t/checkpoints/best.ckpt
    python $FILE  \
                                                    --name $EXPNAME --batch_size 32 --data_batch_size 1 \
                                                    --useSynData  --TEXT --getPairs \
                                                    --ModelName $MODELNAME --pred_trans \
                                                    --ckpt_file $CKPT --data_dir $DATA 
done

## Dessie
MODELNAME=DESSIE
EXPNAME=TOTALRANDOM
for t in 9; do 
    CKPT=$PROJECTPATH/results/model/$EXPNAME/version_$t/checkpoints/best.ckpt
    python $FILE  \
                                                    --name $EXPNAME --batch_size 32 --data_batch_size 1 \
                                                    --useSynData --TEXT \
                                                    --ModelName $MODELNAME --pred_trans \
                                                    --ckpt_file $CKPT --data_dir $DATA 
done