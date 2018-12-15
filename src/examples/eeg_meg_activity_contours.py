import os.path as op
import numpy as np
import os
import mne
import time

from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import meg

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


@utils.timeit
def calc_sample_clusters(args):
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, args.subject, 'meg', 'clusters'))
    labels_fol = utils.make_dir(op.join(MMVT_DIR, args.subject, 'labels'))
    stc_name = 'sample_audvis-meg'
    pick_t = 10
    output_fname = op.join(clusters_root_fol, '{}_contoures_10.pkl'.format(stc_name, pick_t))
    # if op.isfi/le(output_fname) and not args.overwrite:
    #     return
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

    # labels = {hemi: lu.read_labels(args.subject, SUBJECTS_DIR, args.atlas, hemi=hemi, n_jobs=args.n_jobs) for
    #           hemi in utils.HEMIS}
    labels=None

    verts_neighbors_fname = op.join(MMVT_DIR, 'sample', 'verts_neighbors_{}.pkl')
    verts_neighbors_dict = {hemi: utils.load(verts_neighbors_fname.format(hemi)) for hemi in utils.HEMIS}

    from collections import defaultdict
    all_contours = defaultdict(dict)
    thresholds = np.arange(2, 9.6, 1)
    now = time.time()
    for run, threshold in enumerate(thresholds):
        key = '{:.2}'.format(threshold)
        utils.time_to_go(now, run, len(thresholds), 1)
        print('Threshold: {}'.format(threshold))
        threshold_output_fname = op.join(labels_fol, 'clusters-sample_audvis-meg_contours_{:.2f}_{}.npz'.format(threshold, '{hemi}'))
        # if /utils.both_hemi_files_exist(threshold_output_fname):
        #     contours[threshold] = {hemi: utils.Bag(np.load(threshold_output_fname.format(hemi=hemi))) for hemi in utils.HEMIS}
        #     continue
        # utils.delete_folder_files(op.join(clusters_root_fol, 'sample_audvis-meg'))
        contours = meg.find_functional_rois_in_stc(
            args.subject, args.mri_subject, '', stc_name, threshold, threshold_is_precentile=False,
            time_index=pick_t, extract_time_series_for_clusters=False, stc=stc, stc_t_smooth=stc_t_smooth, verts=verts,
            connectivity=connectivity, labels=labels, verts_dict=verts, find_clusters_overlapped_labeles=False,
            verts_neighbors_dict=verts_neighbors_dict, only_contours=True, n_jobs=args.n_jobs)

        for hemi in utils.HEMIS:
            all_contours[key][hemi] = np.where(contours[hemi]['contours']) # {hemi: utils.Bag(np.load(threshold_output_fname.format(hemi=hemi))) for hemi in utils.HEMIS}
        # os.rename(op.join(clusters_root_fol, 'sample_audvis-meg'),
        #           op.join(clusters_root_fol, 'sample_audvis-meg-{:.2f}'.format(threshold)))
        # os.rename(op.join(clusters_root_fol, 'clusters_labels_sample_audvis-meg.pkl'),
        #           op.join(clusters_root_fol, 'clusters_labels_sample_audvis_{:.2f}-meg.pkl'.format(threshold)))
        # for hemi in utils.HEMIS:
        #     os.rename(op.join(labels_fol, 'clusters-sample_audvis-meg_contours_{}.npz'.format(hemi)),
        #               threshold_output_fname.format(hemi=hemi))
    utils.save(all_contours, output_fname)


def join_clusters(args):
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, 'sample', 'meg', 'clusters'))
    stc_name = 'sample_audvis-meg'
    pick_t = 10
    output_fname = op.join(clusters_root_fol, '{}_contoures_10.pkl'.format(stc_name, pick_t))
    if not op.isfile(output_fname):
        return
    contours = utils.load(output_fname)
    thresholds = sorted([float(k) for k in contours.keys()])
    contours_united = {}
    for hemi in utils.HEMIS:
        contours_united[hemi] = dict(contours=np.zeros(contours[thresholds[0]][hemi]['contours'].shape), labels=[], max=0)
        for threshold in thresholds:
            inds = np.where(contours[threshold][hemi]['contours'])
            contours_united[hemi]['contours'][inds] = threshold
    utils.save(contours_united, op.join(clusters_root_fol, '{}_contoures_10_unite.pkl'.format(stc_name, pick_t)))

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