import os.path as op
import numpy as np


def run(mmvt):
    mu = mmvt.utils
    t1_max = 0
    t1_data = {}
    for hemi in mu.HEMIS:
        input_fname = op.join(mu.get_user_fol(), 'surf', 'T1-{}.npy'.format(hemi))
        t1_data[hemi] = np.load(input_fname)
        t1_max = max(np.max(t1_data[hemi]), t1_max)

    mmvt.colorbar.set_colormap('gray')
    mmvt.colorbar.set_colorbar_max_min(t1_max, 0)
    colors_ratio = 256 / t1_max
    for hemi in mu.HEMIS:
        mmvt.coloring.color_hemi_data(hemi, t1_data[hemi], 0, colors_ratio, 0, True)

