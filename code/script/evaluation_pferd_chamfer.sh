PROJECTPATH=/home/x_cili/x_cili_lic/DESSIE
cd $PROJECTPATH/code/src/evalpferd_utils
### remember to add REALPATH
DATA=$PROJECTPATH/data/pferd
FILE=save_all_data.py

## Dessie* 
MODELNAME=DESSIE
EXPNAME=COMBINAREAL
for t in 8 ; do
    CKPT=$PROJECTPATH/results/model/$EXPNAME/version_$t/checkpoints/best.ckpt
    python $FILE  \
                                                    --name $EXPNAME --batch_size 1 --data_batch_size 1 \
                                                    --useSynData --TEXT --getPairs \
                                                    --ModelName $MODELNAME --pred_trans \
                                                    --ckpt_file $CKPT --data_dir $DATA 
done

cd $PROJECTPATH/code/src
python evaluate_chamfer_pferd.py --SAVE --PFERD_results $PROJECTPATH/results/PFERD \
                                                                     --model_dir $PROJECTPATH/code/src/SMAL/smpl_models # --SOTA --PONY