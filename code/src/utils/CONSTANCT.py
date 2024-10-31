import numpy as np

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]# Subtract the ImageNet mean values (0.485, 0.456, 0.406) for channels in RGB format.
IMG_NORM_STD = [0.229, 0.224, 0.225]# Divide by the ImageNet standard deviation values (0.229, 0.224, 0.225) for channels in RGB format.

PLABELSNAME = ['L_Eye','R_Eye', 'Nose',
               'Neck', 'RootOfTail',
               'L_Shoulder', 'L_Elbow', 'L_F_Paw',
               'R_Shoulder', 'R_Elbow', 'R_F_Paw',
               'L_Hip','L_Knee','L_B_Paw','R_Hip','R_Knee','R_B_Paw']
PPOINTINDEX = [ np.array([378]),np.array([1191]),np.array([341, 342, 362, 575, 576, 615, 1154, 1155, 1175]),
               np.array([33, 34, 44, 48, 71, 846, 847, 857, 861, 884]),
               np.array([737, 740, 762, 809]),
               np.array([26, 27, 28, 92, 193]),np.array([9, 100, 102, 103]),np.array([465, 468, 482, 491, 492, 520]),
               np.array([839, 840, 841, 905, 1006]),np.array([822, 913, 915, 916]),np.array([1278, 1281, 1295, 1304, 1305, 1333]),
               np.array([663, 672]),np.array([51]),np.array([422, 451, 455, 528, 529, 530]),
               np.array([1385, 1394]),np.array([864]),np.array([1235, 1264, 1268, 1338, 1339, 1340] )]

TEXTURE_Neworder = \
            { 1: "bay_Thoroughbred",
              2: "palomino_Quarter_Horse",
              3: "chestnut_Morgan",
              4: "buckskin_Tennessee_Walker",
              5: "white_Arabian",
              6: "black_Friesian",
              7: "dapple_Gray_Andalusian",
              8: "pinto_Paint_Horse",}
            
ANIMAL3DPLABELSNAME = ['L_Eye','R_Eye', 'L_Ear', 'R_Ear', 
                       'Nose','Throat', 'RootOfTail','Wither', 
                        'L_Shoulder','R_Shoulder','L_Hip','R_Hip', 
                        'L_F_Paw', 'R_F_Paw','L_B_Paw','R_B_Paw']
ANIMAL3DPOINTINDEX = [np.array([378]),np.array([1191]),np.array([241]),np.array([1054]), #np.array([225,226,401]),np.array([1038,1039,1214]),
               np.array([341, 342, 362, 575, 576, 615, 1154, 1155, 1175]),
               np.array([531,602]), #"Throat"
               np.array([737, 740, 762, 809]), # root_of_tail
               np.array([560, 579]), # Wither
               np.array([26, 27, 28, 92, 193]),np.array([839, 840, 841, 905, 1006]),np.array([663, 672]),np.array([1385, 1394]),
               np.array([465, 468, 482, 491, 492, 520]),np.array([1278, 1281, 1295, 1304, 1305, 1333]),
               np.array([422, 451, 455, 528, 529, 530]),np.array([1235, 1264, 1268, 1338, 1339, 1340])]

QUAD_JOINT_PERM = np.array([
1, 0,
3, 2,
4,
5,
6,
7,
9, 8,
11, 10,
13, 12,
15, 14
])

QUAD_JOINT_NAMES = [
'L_eye',
'R_eye',
'L_ear',
'R_ear',
'Nose',
'Throat',
'Tail',
'Withers',
'L_F_elbow',
'R_F_elbow',
'L_B_elbow',
'R_B_elbow',
'L_F_paw',
'R_F_paw',
'L_B_paw',
'R_B_paw',
]

COMBINAPOINTINDEX = [ np.array([378]),np.array([1191]),np.array([341, 342, 362, 575, 576, 615, 1154, 1155, 1175]),
               np.array([33, 34, 44, 48, 71, 846, 847, 857, 861, 884]),
               np.array([737, 740, 762, 809]),
               np.array([26, 27, 28, 92, 193]),np.array([9, 100, 102, 103]),np.array([465, 468, 482, 491, 492, 520]),
               np.array([839, 840, 841, 905, 1006]),np.array([822, 913, 915, 916]),np.array([1278, 1281, 1295, 1304, 1305, 1333]),
               np.array([663, 672]),np.array([51]),np.array([422, 451, 455, 528, 529, 530]),
               np.array([1385, 1394]),np.array([864]),np.array([1235, 1264, 1268, 1338, 1339, 1340] ),
               #np.array([225,226,401]),np.array([1038,1039,1214]), # L_ears, R_ears
               np.array([241]),np.array([1054]), # L_ears, R_ears
               np.array([531,602]), # "Throat"
               np.array([560, 579]), # Wither
               ]

COMBINAPOINTNAME = ['L_Eye','R_Eye', 'Nose',#0,1,2
               'Neck', 'RootOfTail',#3,4
               'L_Shoulder', 'L_Elbow', 'L_F_Paw',#5,6,7
               'R_Shoulder', 'R_Elbow', 'R_F_Paw',#8,9,10
               'L_Hip','L_Knee','L_B_Paw',#11,12,13,
               'R_Hip','R_Knee','R_B_Paw',#14,15,16
               'L_Ear_inanimal3d', 'R_Ear_inanimal3d', 'Throat_inanimal3d','Wither_in_animal3d',#17,18,19,20
               ]

JOINT_PERM = np.array([1,0,   2,3,4,  8,9,10,5,6,7,  14,15,16,11,12,13])