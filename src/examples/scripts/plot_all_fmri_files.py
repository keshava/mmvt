import glob
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    mmvt.appearance.show_inflated()
    mmvt.coloring.set_lower_threshold(0.95)
    mmvt.colorbar.set_colormap('RdOrYl')
    mmvt.render.save_views_with_cb(True)
    mmvt.render.set_view_distance(22)
    files = glob.glob(op.join(mu.get_user_fol(), 'fmri', 'fmri_*_lh.npy'))
    for fname in files:
        file_name = mu.namebase(fname)[len('fmri_'):-len('_lh')]
        mmvt.coloring.clear_colors()
        mu.center_view()
        mmvt.coloring.set_fmri_file(file_name)
        mmvt.render.set_output_path(mu.make_dir(op.join(mu.get_user_fol(), 'figures', file_name)))
        mmvt.colorbar.set_colorbar_min_max(0.951, 1)
        mmvt.colorbar.set_colorbar_title(file_name.replace('_', ' '))
        mmvt.coloring.plot_fmri()
        mmvt.render.save_all_views()
