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
    create_cubes(mmvt, values, vol_threg, indices, data_min, data_max, vol_name)


def create_cubes(mmvt, values, vol_threg, indices, data_min, data_max, vol_name):
    # https://stackoverflow.com/questions/48818274/quickly-adding-large-numbers-of-mesh-primitives-in-blender
    mu = mmvt.utils
    mmvt.data.create_empty_if_doesnt_exists(vol_name, mmvt.BRAIN_EMPTY_LAYER, None, 'Functional maps')
    orig_cube = mu.create_cube(mmvt.ACTIVITY_LAYER, radius=0.1)
    colors_ratio = 256 / (data_max - data_min)
    colors = mmvt.coloring.calc_colors(values, data_min, colors_ratio)
    now, N = time.time(), len(indices)
    for run, (voxel, ind, color) in enumerate((zip(vol_threg, indices, colors))):
        mu.time_to_go(now, run, N, 100)
        cube_name = 'cube_{}_{}_{}_{}'.format(vol_name[:3], voxel[0], voxel[1], voxel[2])
        cur_obj = mu.get_object(cube_name)
        if cur_obj is not None:
            mu.color_obj(cur_obj.active_material, color)
        else:
            mu.copy_cube(orig_cube, voxel * 0.1, cube_name, vol_name, color)
    mu.delete_current_obj()



