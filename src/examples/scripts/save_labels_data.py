import os.path as op


def run(mmvt):
    mu = mmvt.utils
    for label_data_fname in mmvt.labels.get_labels_data_files():
        label_data_name = mu.namebase(label_data_fname).replace('_', ' ')
        figures_fol = mu.make_dir(op.join(mu.get_user_fol(), 'figures', label_data_name))
        mmvt.coloring.clear_colors()
        mmvt.labels.select_labels_data(label_data_fname)
        mmvt.labels.plot_labels_data()
        mmvt.render.set_output_path(figures_fol)
        mmvt.render.save_views_with_cb()
        mmvt.render.save_all_views()
