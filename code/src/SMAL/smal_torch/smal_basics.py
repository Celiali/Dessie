import os
import pickle as pkl
import numpy as np
import pickle as pkl

import sys, os

def align_smal_template_to_symmetry_axis(model_path, v):
    # These are the indexes of the points that are on the symmetry axis
    I = [522,523,524,531,532,533,534,535,536,537,544,545,546,547,558,559,560,561,
         562,563,564,565,566,567,568,569,570,571,573,574,575,576,577,578,579,580,
         581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,
         599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,639,
         640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,659,
         660,688,689,690,691,692,693,694,695,717,718,719,720,721,728,732,736,740,
         741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,
         759,760,762]

    v = v - np.mean(v)
    y = np.mean(v[I,1])
    v[:,1] = v[:,1] - y
    v[I,1] = 0
    sym_path = os.path.join(os.path.dirname(model_path), 'symmetry_indexes_horse.pkl')
    with open(sym_path, 'rb') as f:
        symIdx = pkl.load(f, encoding="latin1")['symIdx']

    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    v[left[symIdx]] = np.array([1,-1,1])*v[left]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert(len(left_inds) == len(right_inds))
    except:
        import pdb; pdb.set_trace()

    return v, left_inds, right_inds, center_inds, symIdx
