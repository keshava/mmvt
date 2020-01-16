import numpy as np
import mne
import os.path as op
from src.utils import labels_utils as lu
from src.utils import trig_utils as tu
from src.utils import geometry_utils as gu
from src.utils import utils
from tqdm import tqdm

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def prepare_files(subject, remote_subjects_dir):
    necessary_files = {'mri': ['wm.mgz'],
                       'label': ['rh.cortex.label', 'lh.cortex.label'],
                       'surf': ['lh.thickness', 'rh.thickness']}
    utils.prepare_subject_folder(necessary_files, subject, op.join(remote_subjects_dir, subject), SUBJECTS_DIR)


def lables_stat(subject, atlas, excluded=('corpuscallosum', 'unknown')):
    all_labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    labels, _ = lu.remove_exclude_labels(all_labels, excludes=excluded)
    print('Remove {} labels'.format(len(all_labels) - len(labels)))
    vertices_num = [len(l.vertices) for l in labels]
    print('vertivces num: {}-{}'.format(np.min(vertices_num), np.max(vertices_num)))

    labels_area = []
    for l in tqdm(labels):
        vertices, faces = get_vertices_faces(l.vertices, subject, l.hemi)
        labels_area.append(tu.triangle_area(vertices, faces))
    labels_area = np.array(labels_area)
    print('areas: min {}, max {}, mean {}, std {}'.format(
        np.min(labels_area), np.max(labels_area), np.mean(labels_area), np.std(labels_area)))
    fol = utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label', atlas))
    output_fname = op.join(fol, 'labels_stat.npz')
    np.savez(output_fname, labels_area=labels_area, vertices_num=vertices_num)


# @utils.check_for_freesurfer
def fs_stat(subject, atlas, surface='pial', excluded=('corpuscallosum', 'unknown'), overwrite=False):
    labels_area = []
    for hemi in utils.HEMIS:
        output_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.stat'.format(hemi, atlas))
        if not op.isfile(output_fname) or overwrite:
            utils.run_script('mris_anatomical_stats -a {} {} {} {} > {}'.format(
                atlas, subject, hemi, surface, output_fname))
        hemi_labels_names, hemi_labels_area = parse_fs_stat(output_fname, excluded)
        labels_area.extend(hemi_labels_area)
    labels_area = np.array(labels_area)
    print('{}: areas: min {}, max {}, mean {}, std {}'.format(
        subject, np.min(labels_area), np.max(labels_area), np.mean(labels_area), np.std(labels_area)))
    fol = utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label', atlas))
    output_fname = op.join(fol, 'labels_fs_stat.npz')
    np.savez(output_fname, labels_area=labels_area)


def parse_fs_stat(output_fname, excluded):
    from functools import partial
    import re

    def next_index(ind):
        ret = stat.find('structure is "', ind)
        return ret + len('structure is "') if ret != -1 else -1

    labels_area, labels_names = [], []
    _label_is_excluded = partial(
        lu.label_is_excluded, compiled_excludes=re.compile('|'.join(excluded)))

    with open(output_fname, 'r', encoding='utf8', errors='ignore') as input_file:
        stat = input_file.read()
        start_ind = next_index(0)
        while start_ind != -1:
            end_ind = stat.find('"', start_ind)
            label_name = utils.remove_non_printable(stat[start_ind:end_ind])
            if _label_is_excluded(label_name):
                start_ind = next_index(end_ind)
                continue
            labels_names.append(label_name)
            start_ind = stat.find('total surface area', end_ind)
            end_ind = stat.find('mm^2', start_ind)
            label_area = float((utils.find_num_in_str(stat[start_ind:end_ind])[0]))
            labels_area.append(label_area)
            # print('{} area: {} mm^2'.format(label_name, label_area))
            start_ind = next_index(end_ind)
    return labels_names, labels_area


def get_vertices_faces(vertices_ind, subject, hemi, surf='pial'):
    surf = mne.surface.read_surface(op.join(SUBJECTS_DIR, subject, 'surf', hemi + '.' + surf))
    ind = np.in1d(surf[1][:,0], vertices_ind) *\
        np.in1d(surf[1][:,1], vertices_ind) *\
        np.in1d(surf[1][:,2], vertices_ind)
    vertices = surf[0][vertices_ind]
    faces = surf[1][ind]
    faces = np.array([np.where(vertices_ind == face)[0]
                      for face in faces.ravel()]).reshape(faces.shape)
    return vertices, faces




if __name__ == '__main__':
    subjects = ['awmrc_004', 'awmrc_008', 'awmrc_010', 'awmrc_011', 'awmrc_021']
    remote_subjects_dir = '/cluster/fusion/data/resting_state/subjects_mri/'
    atlas = 'laus500'
    for subject in subjects:
        prepare_files(subject, remote_subjects_dir)
        fs_stat(subject, atlas, overwrite=False)
    # lables_stat(subject, atlas)