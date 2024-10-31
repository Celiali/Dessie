#!/bin/bash
PROJECTPATH=/home/x_cili/x_cili_lic/DESSIE
cd $PROJECTPATH/code/src
### remember to add REALPATH
DATA=$PROJECTPATH/data/pferd
FILE=evaluate_pferd.py

## Dessie
MODELNAME=DESSIE
EXPNAME=TOTALRANDOM
for t in 9; do 
    CKPT=$PROJECTPATH/results/model/$EXPNAME/version_$t/checkpoints/best.ckpt
    /home/x_cili/.conda/envs/lassie/bin/python $FILE \
                                                    --name $EXPNAME --batch_size 1 --data_batch_size 1 \
                                                    --useSynData --TEXT --getPairs \
                                                    --ModelName $MODELNAME --pred_trans \
                                                    --ckpt_file $CKPT --data_dir $DATA 
done

cd $PROJECTPATH/code/src
# print pck results
/home/x_cili/.conda/envs/lassie/bin/python evalpferd_utils/cal_average.py --path $PROJECTPATH/results/model/$EXPNAME
# print PA results
/home/x_cili/.conda/envs/lassie/bin/python evalpferd_utils/cal_average.py --path $PROJECTPATH/results/model/$EXPNAME --FLAG