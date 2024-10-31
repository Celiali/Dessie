'''
code adapted from https://github.com/statho/animals3d/blob/main/acsm/constants.py
'''
import numpy as np

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