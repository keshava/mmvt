import os.path as op
from src.utils import utils
from src.preproc import anatomy as anat
from src.preproc import fMRI as fmri

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def calc_contoures(subject, fmri_names, thresholds_min=2, thresholds_max=None, thresholds_dx=2, n_jobs=4):
    if thresholds_max is None:
        thresholds_max = thresholds_min
    for fmri_name in fmri_names:
        fmri.contrast_to_contours(
            subject, fmri_name, thresholds_min, thresholds_max, thresholds_dx,
            min_cluster_size=10, find_clusters_overlapped_labeles=True,
            atlas='aparc.DKTatlas', n_jobs=n_jobs)


def merge_contours(subject, fmri_names):
    for fmri_name in fmri_names:
        pass
        pass

if __name__ == '__main__':
    fmri_names = ['QT_MAP_1_tfce_corrp_tstat1_1mm_insulaopercula',
                  'Inf_MAP_3_tfce_corrp_tstat1_1mm_insulaopercula',
                  'Troponin_MAP_1_tfce_corrp_tstat1_1mm_insulaopercula',
                  'ASH_MAP_3_tfce_corrp_tstat1_1mm_insulaopercula']
    calc_contoures('hbs', fmri_names, 0.95)