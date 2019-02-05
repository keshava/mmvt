import bpy

import os.path as op
import nibabel as nib
import numpy as np
import glob

_mmvt = None


def run(mmvt):
    mu = mmvt.utils
    data_min, data_max = bpy.context.scene.plot_volume_colorbar_min, bpy.context.scene.plot_volume_colorbar_max
    threshold = mmvt.coloring.get_lower_threshold()
    mmvt.coloring.set_lower_threshold(data_min)
    mmvt.colorbar.set_colorbar_min_max(data_min, data_max)
    mmvt.colorbar.set_colormap(bpy.context.scene.plot_volume_colormap_name)
    mmvt.coloring.clear_colors()
    vol_fname = op.join(mu.get_fmri_dir(), mu.get_user(), 'volume', '{}.mgz'.format(bpy.context.scene.plot_volume_fname))
    vol_name = mu.namebase(vol_fname)
    vol = nib.load(vol_fname)
    data = vol.get_data()
    values = data[np.where(data >= threshold)]
    print('{}: {} values above the threshold ({})'.format(vol_name, len(values), threshold))
    indices = np.where(data >= threshold)
    indices = np.array([indices[0], indices[1], indices[2]]).T

    t1 = nib.load(op.join(mu.get_subjects_dir(), mu.get_user(), 'mri', 'T1.mgz'))
    vol_ras = mu.apply_trans(vol.header.get_vox2ras(), indices)
    t1_vox = mu.apply_trans(np.linalg.inv(t1.header.get_vox2ras()), vol_ras)
    t1_tkreg = mu.apply_trans(t1.header.get_vox2ras_tkr(), t1_vox)
    mu.create_cubes(data, values, t1_tkreg, indices, data_min, data_max, vol_name)


def plot_volume_fname_update(self, context):
    if _mmvt is None:
        return
    mu = _mmvt.utils
    vol_fname = op.join(mu.get_fmri_dir(), mu.get_user(), '{}.mgz'.format(bpy.context.scene.plot_volume_fname))
    vol_name = mu.namebase(vol_fname)
    vol = nib.load(vol_fname)
    data = vol.get_data()
    threshold = _mmvt.coloring.get_lower_threshold()
    values = data[np.where(data >= threshold)]
    print('{}: {} values above the threshold ({})'.format(vol_name, len(values), threshold))


bpy.types.Scene.plot_volume_colormap_name = bpy.props.EnumProperty(items=[])
bpy.types.Scene.plot_volume_fname = bpy.props.EnumProperty(items=[]) #bpy.props.StringProperty(subtype='FILE_PATH')
bpy.types.Scene.plot_volume_colorbar_max = bpy.props.FloatProperty()
bpy.types.Scene.plot_volume_colorbar_min = bpy.props.FloatProperty()


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'plot_volume_fname', text='File')
    layout.prop(context.scene, 'plot_volume_colormap_name', text='Colormap')
    row = layout.row(align=0)
    row.prop(context.scene, "plot_volume_colorbar_min", text="min:")
    row.prop(context.scene, "plot_volume_colorbar_max", text="max:")


def init(mmvt):
    _mmvt = mmvt
    mu = mmvt.utils
    colormaps_names = mmvt.colorbar.get_colormaps_names()
    cm_items = [(c, c, '', ind) for ind, c in enumerate(colormaps_names)]
    bpy.types.Scene.plot_volume_colormap_name = bpy.props.EnumProperty(
        items=cm_items, description="colormaps names")
    bpy.context.scene.plot_volume_colormap_name = 'RdOrYl'

    files = sorted([mu.namebase(f) for f in glob.glob(op.join(mu.get_fmri_dir(), mu.get_user(), 'volume', '*.mgz'))])
    files_items = [(c, c, '', ind) for ind, c in enumerate(files)]
    bpy.types.Scene.plot_volume_fname = bpy.props.EnumProperty(
        items=files_items, description="Volume file names", update=plot_volume_fname_update)
    bpy.context.scene.plot_volume_fname = files[0]

    bpy.context.scene.plot_volume_colorbar_min = 0.95
    bpy.context.scene.plot_volume_colorbar_max = 1



