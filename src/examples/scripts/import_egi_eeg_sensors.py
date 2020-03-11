import bpy
import os.path as op
import mne
import numpy as np
from scripts_panel import ScriptsPanel


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    mu = mmvt.mmvt_utils
    input_file = op.join(
        mu.get_user_fol(), 'electrodes',
        '{}.npz'.format(bpy.context.scene.electrodes_positions_files))
    mmvt.data.import_electrodes(
        input_file, mmvt.ELECTRODES_LAYER, overwrite=bpy.context.scene.data_overwrite_electrodes, mult=10)
    bpy.types.Scene.electrodes_imported = True


def fit_fid():
    mmvt = _mmvt()
    # RPA, LPA, Nz
    src_pts = [[7.62704, 0.378298, -4.10108], [-7.62704, 0.378298, -4.10108], [0.0, 8.69387, -3.87288]]
    tgt_pts = [[6.71,-0.07366,-5.538], [-7.4312,-0.1446,-4.258], [-0.4146,7.744,-3.612]]
    eeg_pts = mmvt.electrodes.get_electrodes_pos()
    trans = mne.coreg.fit_matched_points(src_pts, tgt_pts, rotate=True, translate=True, scale=True)
    new_eeg_pts = mne.transforms.apply_trans(trans, eeg_pts, move=True)
    mmvt.electrodes.set_electrodes_pos(new_eeg_pts)
    print('sdf')


def save_eeg_sensors():
    mmvt, mu = _mmvt(), _mmvt().mmvt_utils
    fol = mu.make_dir(op.join(mu.get_user_fol(), 'eeg'))
    eeg_pos_fname = op.join(fol, 'eeg_sensors_positions.npz')
    eeg_names = mmvt.electrodes.get_electrodes_names()
    eeg_pos = mmvt.electrodes.get_electrodes_pos()
    np.savez(eeg_pos_fname, pos=eeg_pos, names=eeg_names)
    print('The EEG sensors were saved to {}'.format(eeg_pos_fname))


def snap_eeg_sensors():
    mmvt, mu = _mmvt(), _mmvt().mmvt_utils
    snap_eeg_sensors_fname = op.join(mu.get_user_fol(), 'eeg', 'eeg_snap_sensors.npz')
    if not op.isfile(snap_eeg_sensors_fname):
        print('You should first run src.preproc.eeg -s subject-name -f snap_sensors_to_outer_skin')
        return
    eeg_info = mu.Bag(np.load(snap_eeg_sensors_fname))
    mmvt.electrodes.set_electrodes_pos(eeg_info.snapped_electrodes / 10)


def draw(self, context):
    layout = self.layout
    layout.operator(EEGFitFid.bl_idname, text="Fit Fiducials", icon='LAMP_AREA')
    layout.operator(EEGSave.bl_idname, text="Save", icon='SAVE_PREFS')
    layout.operator(EEGSnap.bl_idname, text="Snap", icon='SNAP_NORMAL')


class EEGFitFid(bpy.types.Operator):
    bl_idname = "mmvt.eeg_fit_fid"
    bl_label = "mmvt EEG fit fid"
    bl_description = 'Fit EEG to Fid'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        fit_fid()
        return {"FINISHED"}


class EEGSave(bpy.types.Operator):
    bl_idname = "mmvt.eeg_save"
    bl_label = "mmvt EEG save"
    bl_description = 'save EEG sensors'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        save_eeg_sensors()
        return {"FINISHED"}


class EEGSnap(bpy.types.Operator):
    bl_idname = "mmvt.eeg_snap"
    bl_label = "mmvt EEG snap"
    bl_description = 'Snap EEG sensors'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        snap_eeg_sensors()
        return {"FINISHED"}





def init(mmvt):
    register()


def register():
    try:
        bpy.utils.register_class(EEGFitFid)
        bpy.utils.register_class(EEGSave)
        bpy.utils.register_class(EEGSnap)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(EEGFitFid)
        bpy.utils.unregister_class(EEGSave)
        bpy.utils.unregister_class(EEGSnap)
    except:
        pass

