

def run(mmvt):
    # hc034
    mmvt.appearance.show_inflated()
    mmvt.show_hide.hide_hemi('rh')
    mmvt.show_hide.show_hemi('lh')

    # Plot MEG
    mmvt.coloring.clear_colors()
    mmvt.coloring.set_meg_files('dSPM_mean_flip_vertices_power_spectrum_stat')
    mmvt.coloring.set_lower_threshold(1.8)
    mmvt.coloring.color_meg_peak()
    mmvt.colorbar.set_colorbar_max_min(2.15, 1.8)
    mmvt.colorbar.set_colormap('RdOrYl')

    # Plot fMRI left superiorfrontal contours
    mmvt.labels.color_contours(
        specific_hemi='lh', filter='superiorfrontal', specific_colors=[0, 0, 1], atlas='MSIT_I-C', move_cursor=False)