import os.path as op
import numpy as np
import mne
import time
from collections import defaultdict

from src.utils import utils
from src.preproc import meg

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


@utils.timeit
def calc_sample_clusters(args):
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, args.subject, 'meg', 'clusters'))
    stc_name = 'sample_audvis-meg'
    pick_t = 10
    output_fname = op.join(clusters_root_fol, '{}_contoures_10.pkl'.format(stc_name, pick_t))
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

    verts_neighbors_fname = op.join(MMVT_DIR, 'sample', 'verts_neighbors_{}.pkl')
    verts_neighbors_dict = {hemi: utils.load(verts_neighbors_fname.format(hemi)) for hemi in utils.HEMIS}

    all_contours = {}
    thresholds = np.arange(2, 11, 1)
    now = time.time()
    for run, threshold in enumerate(thresholds):
        key = '{:.2f}'.format(threshold)
        all_contours[key] = {}
        utils.time_to_go(now, run, len(thresholds), 1)
        flag, contours = meg.find_functional_rois_in_stc(
            args.subject, args.mri_subject, '', stc_name, threshold, threshold_is_precentile=False,
            time_index=pick_t, extract_time_series_for_clusters=False, stc=stc, stc_t_smooth=stc_t_smooth, verts=verts,
            connectivity=connectivity, verts_dict=verts, find_clusters_overlapped_labeles=False,
            verts_neighbors_dict=verts_neighbors_dict, only_contours=True, save_results=False, n_jobs=args.n_jobs)
        for hemi in utils.HEMIS:
            all_contours[key][hemi] = np.where(contours[hemi]['contours'])
    utils.save(all_contours, output_fname)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, default='sample')
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='sample')
    parser.add_argument('-f', '--function', help='function name', required=False, default='calc_sample_clusters')
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
    print('Done!')