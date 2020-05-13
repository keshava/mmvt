import os.path as op
import numpy as np
import bpy


# hc034
def run(mmvt):
    mu = mmvt.utils

    # Appearance
    mmvt.appearance.set_inflated_ratio(-0.38)
    mmvt.show_hide.hide_hemi('rh')
    mmvt.show_hide.show_hemi('lh')

    # Plot MEG
    mmvt.coloring.clear_colors()
    mmvt.coloring.set_meg_files('dSPM_mean_flip_vertices_power_spectrum_stat')
    mmvt.coloring.set_lower_threshold(1.7)
    mmvt.coloring.color_meg_peak()
        mmvt.colorbar.set_colorbar_max_min(2.15, 1.7)
        mmvt.colorbar.set_colormap('RdOrYl')
        mmvt.colorbar.set_colorbar_title('MEG power -log10(pval)')

    # Plot fMRI left superiorfrontal contours
    mmvt.labels.color_contours(
        specific_hemi='lh', filter='superiorfrontal', specific_colors=[0, 0, 1], atlas='MSIT_I-C', move_cursor=True)

    # Add the power-spectrum data to the graph panel
    input_fname = op.join(mu.get_user_fol(), 'meg', 'labels_data_MSIT_power_spectrum_stat_lh.npz')
    if not op.isfile(input_fname):
        print('No data file! {}'.format(input_fname))
    d = mu.Bag(np.load(input_fname))
    mmvt.data.add_data_pool('meg_power_spectrum_stat', d.data, d.conditions)
    bpy.data.objects['meg_power_spectrum_stat'].select = True
    mmvt.selection.fit_selection()
    for marker_ind, marker_name in [(245, '71Hz'), (123, '36Hz'), (85, '25Hz'), (384, '110Hz'), (271, '78Hz')]:
        mu.add_marker(marker_ind, marker_name)
    mmvt.coloring.set_current_time(269)
