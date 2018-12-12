import os.path as op
import numpy as np
import os

from src.utils import utils
from src.preproc import meg

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def calc_sample_clusters(args):
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
            threshold=threshold,
            threshold_is_precentile=False,
            min_cluster_size=100,
        ))
        meg.call_main(_args)
        os.rename(op.join(clusters_root_fol, 'sample_audvis-meg'),
                  op.join(clusters_root_fol, 'sample_audvis-meg-{:.2f}'.format(threshold)))
        os.rename(op.join(clusters_root_fol, 'clusters_labels_sample_audvis-meg.pkl'),
                  op.join(clusters_root_fol, 'clusters_labels_sample_audvis_{:.2f}-meg.pkl'.format(threshold)))



if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    from src.utils import preproc_utils as pu

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='')
    parser.add_argument('-a', '--atlas', required=False, default='laus125')
    parser.add_argument('-t', '--task', required=False, default='MSIT')
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.subject = pu.decode_subjects(args.subject, remote_subject_dir=args.remote_subject_dir)
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
    print('Done!')