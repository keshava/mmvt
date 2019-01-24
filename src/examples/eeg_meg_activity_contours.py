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
    stc_name = 'sample_audvis-meg'
    pick_t = 10
    thresholds_min, thresholds_max, thresholds_dx = 2, 11, 1
    meg.stc_to_contours(args.subject, stc_name, pick_t, thresholds_min, thresholds_max, thresholds_dx)


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