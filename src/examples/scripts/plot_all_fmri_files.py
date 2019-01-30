import glob
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    # mmvt.appearance.show_inflated()
    mmvt.appearance.show_pial()
    mmvt.show_hide.hide_subcorticals()
    mmvt.coloring.set_lower_threshold(0.95)
    mmvt.colorbar.set_colormap('YlOrRd')
    mmvt.colorbar.set_colorbar_title('p-vals')
    # mmvt.render.save_views_with_cb(True)
    # mmvt.render.set_view_distrecibnance(22)
    files = glob.glob(op.join(mu.get_user_fol(), 'fmri', 'fmri_*_lh.npy'))
    mmvt.transparency.set_light_layers_depth(2)
    for fname in files:
        file_name = mu.namebase(fname)[len('fmri_'):-len('_lh')]
        mmvt.coloring.clear_colors()
        mu.center_view()
        mmvt.coloring.set_fmri_file(file_name)
        mmvt.render.set_output_path(mu.make_dir(op.join(mu.get_user_fol(), 'figures', file_name)))
        mmvt.colorbar.set_colorbar_min_max(0.951, 1)
        mmvt.colorbar.set_colorbar_title(file_name.replace('_', ' '))
        mmvt.coloring.plot_fmri()
        mmvt.render.save_all_views(render_images=True, quality=60)

