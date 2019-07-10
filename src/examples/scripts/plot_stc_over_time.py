import bpy
import numpy as np
import os.path as op
import mne
from scripts_panel import ScriptsPanel
from play_panel import GrabFromPlay, GrabToPlay
from tqdm import tqdm


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    mu = mmvt.utils
    stc_fname = mmvt.coloring.get_stc_full_fname()
    if not op.isfile(stc_fname):
        print('Can\'t find the selected stc!')
        return
    stc = mne.read_source_estimate(stc_fname)
    data = {}
    if mmvt.play.get_play_to() > len(stc.times) - 1:
        mmvt.play.set_play_to(len(stc.times) - 1)
    time = np.arange(mmvt.play.get_play_from(), mmvt.play.get_play_to(), mmvt.play.get_play_dt())
    data['rh'] = np.zeros((stc.rh_data.shape[0], 1))
    data['lh'] = np.zeros((stc.lh_data.shape[0], 1))
    threshold = mmvt.coloring.get_lower_threshold()
    for t_ind, t in tqdm(enumerate(time)):
        for hemi in mu.HEMIS:
            hemi_data = stc.rh_data[:, t_ind] if hemi=='rh' else stc.lh_data[:, t_ind]
            verts = np.where(hemi_data >= threshold)[0]
            data[hemi][verts, 0] = time[t_ind]

    data = np.concatenate([data['lh'], data['rh']])
    vertices = [stc.lh_vertno, stc.rh_vertno]
    stc = mne.SourceEstimate(data, vertices, 0, 0, subject=mu.get_user())
    mmvt.colorbar.lock_colorbar_values(False)
    mmvt.coloring.clear_colors()
    mmvt.coloring.plot_stc(stc, bpy.context.scene.frame_current, 0, time[-1], time[0])
    mmvt.coloring.set_lower_threshold(threshold) # Set threshold to its previous value


def draw(self, context):
    layout = self.layout
    row = layout.row(align=0)
    row.prop(context.scene, "play_from", text="From")
    row.operator(GrabFromPlay.bl_idname, text="", icon='BORDERMOVE')
    row.prop(context.scene, "play_to", text="To")
    row.operator(GrabToPlay.bl_idname, text="", icon='BORDERMOVE')
    # row = layout.row(align=0)
    layout.prop(context.scene, "play_dt", text="dt")
