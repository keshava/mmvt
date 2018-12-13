import os.path as op
import numpy as np
import os
import mne

from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import meg

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def calc_sample_clusters(args):
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, args.subject, 'meg', 'clusters'))
    labels_fol = utils.make_dir(op.join(MMVT_DIR, args.subject, 'labels'))
    stc_name = 'sample_audvis-meg'
    pick_t = 10
    connectivity = meg.load_connectivity(args.subject)
    stc_t_smooth_fname = op.join(clusters_root_fol, '{}_{}_smooth'.format(stc_name, pick_t))
    stc_fname = op.join(MMVT_DIR, args.subject, 'meg', '{}-lh.stc'.format(stc_name))
    if not op.isfile(stc_fname):
        raise Exception("Can't find the stc file! ({})".format(stc_name))
    stc = mne.read_source_estimate(stc_fname)
    if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_t_smooth_fname, '{hemi}')):
        stc_t_smooth = mne.read_source_estimate(stc_t_smooth_fname)
        verts = meg.get_pial_vertices(args.subject)
    else:
        stc_fname = op.join(MMVT_DIR, args.subject, 'meg', '{}-lh.stc'.format(stc_name))
        if not op.isfile(stc_fname):
            raise Exception("Can't find the stc file! ({})".format(stc_name))
        stc = mne.read_source_estimate(stc_fname)
        stc_t = meg.create_stc_t(stc, pick_t, args.subject)
        stc_t_smooth = meg.calc_stc_for_all_vertices(stc_t, args.subject, args.subject, args.n_jobs)
        stc_t_smooth.save(stc_t_smooth_fname)
        verts = meg.check_stc_with_ply(stc_t_smooth, subject=args.subject)

    labels = {hemi: lu.read_labels(args.subject, SUBJECTS_DIR, args.atlas, hemi=hemi, n_jobs=args.n_jobs) for
              hemi in utils.HEMIS}
    for threshold in np.arange(2, 9.6, 0.1):
        utils.delete_folder_files(op.join(clusters_root_fol, 'sample_audvis-meg'))
        meg.find_functional_rois_in_stc(
            args.subject, args.mri_subject, args.atlas, stc_name, threshold, threshold_is_precentile=False,
            time_index=pick_t, extract_time_series_for_clusters=False, stc=stc, stc_t_smooth=stc_t_smooth, verts=verts,
            connectivity=connectivity, labels=labels, n_jobs=args.n_jobs)

        os.rename(op.join(clusters_root_fol, 'sample_audvis-meg'),
                  op.join(clusters_root_fol, 'sample_audvis-meg-{:.2f}'.format(threshold)))
        os.rename(op.join(clusters_root_fol, 'clusters_labels_sample_audvis-meg.pkl'),
                  op.join(clusters_root_fol, 'clusters_labels_sample_audvis_{:.2f}-meg.pkl'.format(threshold)))
        for hemi in utils.HEMIS:
            os.rename(op.join(labels_fol, 'clusters-sample_audvis-meg_contours_{}.npz'.format(hemi)),
                      op.join(labels_fol, 'clusters-sample_audvis-meg_contours_{:.2f}_{}.npz'.format(threshold, hemi)))


def join_clusters(args):
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, 'sample', 'meg', 'clusters'))
    for threshold in np.arange(2, 9.6, 0.1):
        clusters_dict = utils.Bag(utils.load(
            op.join(clusters_root_fol, 'clusters_labels_sample_audvis_{:.2f}-meg.pkl'.format(threshold))))
        print('asdf')


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    from src.utils import preproc_utils as pu

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, default='sample')
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='sample')
    # parser.add_argument('-a', '--atlas', required=False, default='laus125')
    # parser.add_argument('-t', '--task', required=False, default='MSIT')
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
    print('Done!')