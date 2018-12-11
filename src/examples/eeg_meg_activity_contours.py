import os.path as op
from src.utils import utils
from src.preproc import meg

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def calc_sample_clusters(args):
    import numpy as np
    import os

    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, 'sample', 'meg', 'clusters'))
    for threshold in np.arange(2, 9.6, 0.1):
        utils.delete_folder_files(op.join(clusters_root_fol, 'sample_audvis-meg'))
        _args = meg.read_cmd_args(dict(
            subject=args.subject,
            mri_subject=args.mri_subject,
            task='audvis',
            conditions=['LA', 'RA'],
            function='find_functional_rois_in_stc',
            stc_name='sample_audvis-meg',
            inv_fname='sample_audvis-meg-eeg-oct-6-meg-eeg-inv',
            label_name_template='*',
            peak_stc_time_index=10,
            # peak_mode='pos',
            threshold=threshold,#99.5,
            threshold_is_precentile=False,
            # min_cluster_max=5,
            min_cluster_size=100,
            # recreate_src_spacing='ico5'
            # clusters_label='precentral'
        ))
        meg.call_main(_args)
        os.rename(op.join(clusters_root_fol, 'sample_audvis-meg'),
                  op.join(clusters_root_fol, 'sample_audvis-meg-{:.2f}'.format(threshold)))
        os.rename(op.join(clusters_root_fol, 'clusters_labels_sample_audvis-meg.pkl'),
                  op.join(clusters_root_fol, 'clusters_labels_sample_audvis_{:.2f}-meg.pkl'.format(threshold)))

