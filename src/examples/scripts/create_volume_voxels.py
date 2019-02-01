import bpy
from mathutils import Vector

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
    vox2tkreg = vol.header.get_vox2ras_tkr()
    indices = np.where(data >= mmvt.coloring.get_lower_threshold())
    indices = np.array([indices[0], indices[1], indices[2]]).T
    vol_threg = mu.apply_trans(vox2tkreg, indices)
    mmvt.data.create_empty_if_doesnt_exists(vol_name, mmvt.BRAIN_EMPTY_LAYER, None, 'Functional maps')
    layers = [False] * 20
    layers[mmvt.ACTIVITY_LAYER] = True
    colors_ratio = 256 / (data_max - data_min)
    create_cubes(mmvt, data, vol_threg, indices, data_min, colors_ratio, vol_name, layers)


def create_cubes(mmvt, data, vol_threg, indices, data_min, colors_ratio, vol_name, layers):
    # https://stackoverflow.com/questions/48818274/quickly-adding-large-numbers-of-mesh-primitives-in-blender
    mu = mmvt.utils
    bpy.ops.mesh.primitive_cube_add(radius=0.1)
    bpy.ops.object.move_to_layer(layers=layers)
    orig_cube = bpy.context.active_object

    values = data[np.where(data >= mmvt.coloring.get_lower_threshold())]
    colors = mmvt.coloring.calc_colors(values, data_min, colors_ratio)

    now, N = time.time(), len(indices)
    for run, (voxel, ind, color) in enumerate((zip(vol_threg, indices, colors))):
        mu.time_to_go(now, run, N, 100)
        cube_name = 'cube_{}_{}_{}_{}'.format(vol_name[:3], voxel[0], voxel[1], voxel[2])
        cur_obj = bpy.data.objects.get(cube_name, None)
        if cur_obj is not None:
            color_obj(cur_obj.active_material, color)
            continue

        m = orig_cube.data.copy()
        cur_obj = bpy.data.objects.new('cube', m)
        cur_obj.name = cube_name
        cur_mat = bpy.data.materials['Deep_electrode_mat'].copy()
        cur_mat.name = cur_obj.name + '_Mat'
        cur_obj.active_material = cur_mat
        cur_obj.location = Vector(tuple(voxel * 0.1))
        cur_obj.parent = bpy.data.objects[vol_name]
        color_obj(cur_mat, color)
        bpy.context.scene.objects.link(cur_obj)

    bpy.ops.object.delete()


def color_obj(cur_mat, color):
    cur_mat.node_tree.nodes["RGB"].outputs[0].default_value = (color[0], color[1], color[2], 1)
    cur_mat.diffuse_color = color[:3]
