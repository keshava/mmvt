import os.path as op
import glob
import numpy as np
import mne
from tqdm import tqdm
from src.utils import utils
from src.preproc import anatomy as anat
from src.preproc import fMRI as fmri

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def project_all_fmri_files(subject, fmri_names):
    for fmri_name in fmri_names:
        fmri_fname = op.join(FMRI_DIR, subject, 'volume', '{}.mgz'.format(fmri_name))
        fmri.direct_project_volume_to_surf(subject, fmri_fname, flip_x=True, overwrite=True)


def calc_contoures(subject, fmri_names, thresholds_min=2, thresholds_max=None, thresholds_dx=2, n_jobs=4):
    if thresholds_max is None:
        thresholds_max = thresholds_min
    for fmri_name in fmri_names:
        fmri.contrast_to_contours(
            subject, fmri_name, thresholds_min, thresholds_max, thresholds_dx,
            min_cluster_size=10, find_clusters_overlapped_labeles=True,
            atlas='aparc.DKTatlas', n_jobs=n_jobs)


def merge_labels(subject, fmri_names):
    vertices_labels_lookup = utils.load(op.join(MMVT_DIR, subject, 'aparc.DKTatlas40_vertices_labels_lookup.pkl'))
    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'fmri', 'labels'))
    for fmri_name in fmri_names:
        labels = []
        for hemi in utils.HEMIS:
            surf_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}.npy'.format(fmri_name.replace('_insulaopercula', ''), hemi))
            surf_data = np.load(surf_fname)
            vertices_indices = np.where(surf_data >= 0.95)[0]
            if len(vertices_indices) == 0:
                continue
            insulaopercula_vertices = []
            vertices, _ = utils.read_pial(subject, MMVT_DIR, hemi)
            for vert_ind in tqdm(vertices_indices):
                vert_label = vertices_labels_lookup[hemi][vert_ind]
                if vert_label.startswith('insula'): # or vert_label.startswith('parsopercularis')
                    insulaopercula_vertices.append(vert_ind)
            label = mne.Label(
                insulaopercula_vertices, vertices[insulaopercula_vertices], hemi=hemi, name=fmri_name, subject=subject)
            labels.append(label)
            label.save(op.join(output_fol, '{}.label'.format(fmri_name)))
        anat.labels_to_annot(subject, atlas=fmri_name, labels=labels)
        anat.calc_labeles_contours(subject, fmri_name)


if __name__ == '__main__':
    subject = 'hbs'
    fmri_names = ['QT_MAP_1_tfce_corrp_tstat1_1mm', #_insulaopercula',
                  'Inf_MAP_3_tfce_corrp_tstat1_1mm', #_insulaopercula',
                  'Troponin_MAP_1_tfce_corrp_tstat1_1mm', #_insulaopercula',
                  'ASH_MAP_3_tfce_corrp_tstat1_1mm'] #_insulaopercula']
    # project_all_fmri_files(subject, fmri_names)
    # calc_contoures(subject, fmri_names, 0.95)
    merge_labels(subject, fmri_names)