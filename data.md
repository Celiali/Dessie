## Data preparation 

### For DessiePIPE
1. Please download the uvmaps at this [link](https://drive.google.com/file/d/1GNKv7AKyeLoa5K6KlFvKYvophiWuptjv/view?usp=sharing) derived from [TEXTure](https://texturepaper.github.io/TEXTurePaper/) and place it under `./code/src/SMAL` folder.


2. Download the data at this [link](https://drive.google.com/drive/folders/1eD_16E-tiFisxjIc5xaZyW0lAxqP9FY9?usp=sharing) and place it under `./data/syndata` folder.

- **For ```syndata/coco/```**: we use the coco2017 dataset

- **For ```syndata/pose/```**: we provide pose data derived from [PFERD](https://github.com/Celiali/PFERD).

- **For ```syndata/TEXTure/```**: we provide the texture maps derived from [TEXTure](https://texturepaper.github.io/TEXTurePaper/). 


### For real dataset 

Download the data and place it under `./data` folder.
- **For ```magicpony/```**: Please download the original dataset from [MagicPony](https://github.com/elliottwu/MagicPony/blob/main/data/download_horse_combined.sh) and we provide the pseudo keypoint at this [link](https://drive.google.com/file/d/1kDB_KcbOkr7Vx5qDPMlU878CqTG-cugL/view?usp=sharing) extracted from [ViTPose+](https://github.com/ViTAE-Transformer/ViTPose).

- **For ```staths/```**: Please download the data following instructions in [Staths](https://github.com/statho/animals3d), the data would include the raw AnimalPose dataset and the raw Pascal dataset. 
    1) Download AnimalPose dataset [link](https://drive.google.com/drive/folders/1xxm6ZjfsDSmv6C9JvbgiGrmHktrUjV5x)
    2) Download Pascal dataset [link](https://github.com/statho/animals3d/blob/main/prepare_data/download_pascal.sh)
    3) Download annotation files in [link](https://drive.google.com/file/d/14NTnURgs2RX2WNJIFeSt0fCfzl5zxdBj/view?usp=sharing), extract and move the ```cachedir/data/``` under ```staths/```

- **For ```realimg/```**: We provide the images and annotations this [link](https://drive.google.com/file/d/1NjtF65Q6uSjNrpaOnNKzP8042-c-19pM/view?usp=sharing) derived from [Staths](https://github.com/statho/animals3d). We call data from Staths as `Animal3D` in the code given their repo named as `animals3d`.

- **For ```pferd/```**: Please download data from [link](https://doi.org/10.7910/DVN/2EXONE)

### The directory structure of the `./code/src/SMAL` is as below:
```
|-- src/SMAL
    |--smpl_models
    |--uvmap
```

### The directory structure of `./data` is as below. 

```
|-- data
    |-- syndata (for Dessiepipe)
        |-- coco 
        |-- pose 
        |-- TEXTure 
    |-- realimg (for finetune and evaluation)
        |-- coco_pascal
            |-- annotations
            |-- images
        |-- pascal_val
            |-- annotations
            |-- images  
    |--magicpony (for finetune and evaluation)
        |-- train
        |-- test
        |-- val
    |-- staths (for evaluation)
        |-- animal_pose
            |-- images
        |-- pascal 
            |-- images
        |-- data
            |-- animal_pose
                |-- annotations
                |-- filelists
            |-- pascal
                |-- annotatins
                |-- filelists
    |-- pferd (for evaluation)
        |-- ID1
            |-- CAM_DATA
            |-- KP2D_DATA
            |-- MODEL_DATA
            |-- VIDEO_DATA
        |-- ID2
            |-- CAM_DATA
            |-- KP2D_DATA
            |-- MODEL_DATA
            |-- VIDEO_DATA
        |-- ID4
            |-- CAM_DATA
            |-- KP2D_DATA
            |-- MODEL_DATA
            |-- VIDEO_DATA

```