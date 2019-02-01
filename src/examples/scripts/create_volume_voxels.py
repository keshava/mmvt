import os.path as op
import nibabel as nib
import numpy as np
import time


def run(mmvt):
    mu = mmvt.utils
    data_min, data_max = 0.95, 1
    mmvt.coloring.set_lower_threshold(data_min)
    mmvt.colorbar.set_colorbar_min_max(data_min, data_max)
    # mmvt.colorbar.set_colormap('YlOrRd')
    mmvt.coloring.clear_colors()
    vol_fname = op.join(mu.get_fmri_dir(), mu.get_user(), 'volume', 'Troponin_MAP_1_tfce_corrp_tstat1_1mm_insulaopercula.mgz')
    vol_name = mu.namebase(vol_fname)
    vol = nib.load(vol_fname)
    data = vol.get_data()
    values = data[np.where(data >= mmvt.coloring.get_lower_threshold())]
    vox2tkreg = vol.header.get_vox2ras_tkr()
    indices = np.where(data >= mmvt.coloring.get_lower_threshold())
    indices = np.array([indices[0], indices[1], indices[2]]).T
    vol_threg = mu.apply_trans(vox2tkreg, indices)
    mu.create_cubes(values, vol_threg, indices, data_min, data_max, vol_name)




