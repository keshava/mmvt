import bpy
import os.path as op
import nibabel as nib
import numpy as np


def run(mmvt):
    mu = mmvt.utils
    vol_fname = op.join(mu.get_fmri_dir(), mu.get_user(), 'volume', 'Troponin_MAP_1_tfce_corrp_tstat1_1mm_insulaopercula.mgz')
    vol = nib.load(vol_fname)
    data = vol.get_data()
    vox2tkreg = vol.header.get_vox2ras_tkr()
    indices = np.where(data > mmvt.coloring.get_lower_threshold())
    indices = np.array([indices[0], indices[1], indices[2]]).T
    vol_threg = mu.apply_trans(vox2tkreg, indices)
    for voxel in vol_threg[:50]:
        bpy.ops.mesh.primitive_cube_add(radius=0.1, location=voxel * 0.1, layers=my_layers)
        bpy.ops.object.shade_smooth()
