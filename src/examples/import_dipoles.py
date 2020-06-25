import os.path as op
from collections import defaultdict
from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import meg
import mne
import numpy as np

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def plot_dipole(dip_fname, subject):
    import matplotlib.pyplot as plt
    dips = mne.dipole.read_dipole(dip_fname)
    trans_file = meg.find_trans_file(subject=subject)
    mode = 'orthoview'
    if mode == 'arrow':
        dips.plot_locations(trans_file, subject, SUBJECTS_DIR, mode='arrow')
        from mayavi import mlab
        mlab.show()
    else:
        dips[0].plot_locations(trans_file, subject, SUBJECTS_DIR, mode='orthoview')
        plt.show()


def parse_dip_file(dip_fname):
    '''
    Parse dip file and return dict of events
    :param dip_fname:
    :return: dict of events
    '''
    # Can take code from mne.dipole.read_dipole(dip_fname)
    # read_dipole does not take the dipoles' names
    same_name_lines = []
    results = defaultdict(list)
    line_start = len("## Name \"")
    with open(dip_fname , 'r') as target:
        for line in target.readlines():
            if( line.startswith('#')):
                if same_name_lines:
                    tmp_line = (line[line_start:])
                    name =  (tmp_line.split("\"", maxsplit=1)[0])
                    for item in same_name_lines:
                        results[name].append([float(x) for x in item.split()])
                    same_name_lines = []
            else:
                same_name_lines.append(line)

    return results
    # events = defaultdict(list)
    # # Should read the whole dip file in loop to find all the events
    # event = [162023.8, 162023.8, 30.7, 63.7, 69.6, 89.84, 69.58, -50.12, 26.78, 77.3]
    # events['run2_162'].append(event)
    # return events


def convert_dipoles_to_mri_space(subject, dipoles, overwrite=False):
    '''
    :param dipole:
    :return:
    '''
    output_fname = op.join(utils.make_dir(op.join(MMVT_DIR, subject, 'meg')), 'dipoles.pkl')
    if op.isfile(output_fname) and not overwrite:
        return True
    # If the trans file doesn't exist, you should calculate it using mne-python / MNE-analyzer
    trans_file = meg.find_trans_file(subject=subject)
    head_mri_trans = mne.transforms.read_trans(trans_file)
    head_mri_trans = mne.transforms._ensure_trans(head_mri_trans, 'head', 'mri')

    mri_dipoles = defaultdict(list)
    for dipole_name, dipoles in dipoles.items():
        for dipole in dipoles:
            # begin end(ms)  X (mm)  Y (mm)  Z (mm)  Q(nAm) Qx(nAm) Qy(nAm) Qz(nAm)  g(%)
            begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            mri_pos = mne.transforms.apply_trans(head_mri_trans, [np.array([x, y, z]) * 1e-3])[0]
            dir_xyz = mne.transforms.apply_trans(head_mri_trans, [np.array([qx, qy, qz]) / q])[0]
            print('{}: loc:{} dir:{}'.format(dipole_name, mri_pos, dir_xyz))
            mri_dipoles[dipole_name].append([begin_t, end_t, *mri_pos, q, *dir_xyz, gf])
    print('Saving dipoles in {}'.format(output_fname))
    utils.save(mri_dipoles, output_fname)
    return op.isfile(output_fname)


def calc_dipoles_rois(subject, atlas='laus125', overwrite=False, n_jobs=4):
    links_dir = utils.get_links_dir()
    subjects_dir = utils.get_link_dir(links_dir, 'subjects')
    mmvt_dir = utils.get_link_dir(links_dir, 'mmvt')
    diploes_rois_output_fname = op.join(mmvt_dir, subject, 'meg', 'dipoles_rois.pkl')
    if op.isfile(diploes_rois_output_fname) and not overwrite:
        diploes_rois = utils.load(diploes_rois_output_fname)
        for dip in diploes_rois.keys():
            diploes_rois[dip]['cortical_probs'] *= 1/sum(diploes_rois[dip]['cortical_probs'])
            diploes_rois[dip]['subcortical_probs'] = []
            diploes_rois[dip]['subcortical_rois'] = []
        # coritcal_labels = set(utils.flat_list_of_lists([diploes_rois[k]['cortical_rois'] for k in diploes_rois.keys()]))
        utils.save(diploes_rois, diploes_rois_output_fname)
        return True

    diploes_input_fname = op.join(mmvt_dir, subject, 'meg', 'dipoles.pkl')
    if not op.isfile(diploes_input_fname):
        print('No dipoles file!')
        return False

    labels = lu.read_labels(subject, subjects_dir, atlas, n_jobs=n_jobs)
    labels = list([{'name': label.name, 'hemi': label.hemi, 'vertices': label.vertices}
                   for label in labels])
    if len(labels) == 0:
        print('Can\'t find the labels for atlas {}!'.format(atlas))
        return False

    # find the find_rois package
    mmvt_code_fol = utils.get_mmvt_code_root()
    ela_code_fol = op.join(utils.get_parent_fol(mmvt_code_fol), 'electrodes_rois')
    if not op.isdir(ela_code_fol) or not op.isfile(op.join(ela_code_fol, 'find_rois', 'main.py')):
        print("Can't find ELA folder!")
        print('git pull https://github.com/pelednoam/electrodes_rois.git')
        return False

    # load the find_rois package
    try:
        import sys
        if ela_code_fol not in sys.path:
            sys.path.append(ela_code_fol)
        from find_rois import main as ela
    except:
        print('Can\'t load find_rois package!')
        utils.print_last_error_line()
        return False

    dipoles_dict = utils.load(diploes_input_fname)
    diploles_names, dipoles_pos = [], []
    for cluster_name, dipoles in dipoles_dict.items():
        for begin_t, _, x, y, z, _, _, _, _, _ in dipoles:
            dipole_name = '{}_{}'.format(cluster_name, begin_t) if len(dipoles) > 1 else cluster_name
            diploles_names.append(dipole_name.replace(' ', ''))
            dipoles_pos.append([k * 1e3 for k in [x, y, z]])
    dipoles_rois = ela.identify_roi_from_atlas(
        atlas, labels, diploles_names, dipoles_pos, approx=3, elc_length=0, hit_only_cortex=True,
        subjects_dir=subjects_dir, subject=subject, n_jobs=n_jobs)
    # Convert the list to a dict
    dipoles_rois_dict = {dipoles_rois['name']: dipoles_rois for dipoles_rois in dipoles_rois}
    utils.save(dipoles_rois_dict, diploes_rois_output_fname)


def calc_distances_from_rois(subject, dist_threshold=0.05):
    from scipy.spatial.distance import cdist
    import nibabel as nib
    dipoles_dict = utils.load(op.join(MMVT_DIR, subject, 'meg', 'dipoles.pkl'))
    labels_times_fol = op.join(MMVT_DIR, subject, 'meg', 'time_accumulate')
    labels = lu.read_labels(subject, SUBJECTS_DIR, 'laus125')
    labels_center_of_mass = lu.calc_center_of_mass(labels)
    labels_pos = np.array([labels_center_of_mass[l.name] for l in labels])
    labels_dict = {l.name: labels_center_of_mass[l.name] for l in labels}
    outer_skin_surf_fname = op.join(SUBJECTS_DIR, subject, 'surf', 'lh.seghead')
    outer_skin_surf_verts, _ = nib.freesurfer.read_geometry(outer_skin_surf_fname)

    for dipole_name, dipoles in dipoles_dict.items():
        dipole_pos = np.array([dipoles[0][2], dipoles[0][3], dipoles[0][4]])
        lables_times_fname = op.join(labels_times_fol, '{}_labels_times.txt'.format(dipole_name))
        if not op.isfile(lables_times_fname):
            print('Can\'t find {}!'.format(lables_times_fname))
            continue
        dists_from_outer_skin = np.min(cdist(outer_skin_surf_verts * 0.001, [dipole_pos]), 0)[0]
        output_fname = op.join(labels_times_fol, '{}_labels_times_dists.txt'.format(dipole_name))
        lines = utils.csv_file_reader(lables_times_fname, delimiter=':', skip_header=1)
        output, dists = [], []
        labels_dists = cdist(labels_pos, [dipole_pos])
        dists_argmin = np.argmin(labels_dists, 0)[0]
        dists_min = np.min(labels_dists, 0)[0]
        closest_label = labels[dists_argmin].name
        print('Parsing {} ({})'.format(dipole_name, closest_label))
        for line in lines:
            if len(line) == 0:
                continue
            elif len(line) != 2:
                print('{}: Problem parsing "{}"'.format(lables_times_fname, line))
                continue
            label_name, label_time = line
            label_pos = labels_dict.get(label_name, None)
            if label_pos is not None:
                dist_from_dipole = np.linalg.norm(dipole_pos - label_pos)
                dists.append(dist_from_dipole)
            else:
                dist_from_dipole = -1
                dists.append(np.nan)
            output.append('{}: {} ({:.4f})'.format(label_name, label_time, dist_from_dipole))
        for ind, dist in enumerate(dists):
            if dist < dist_threshold:
                output[ind] = '{} ***'.format(output[ind])
        title = '{}: {} {:.4f} dist from outer skin: {:.4f} '.format(
            dipole_name, closest_label, dists_min, dists_from_outer_skin)
        utils.save_arr_to_file(output, output_fname, title)


if __name__ == '__main__':
    subject = 'nmr01426'# 'nmr01391'
    atlas = 'aparc.DKTatlas40' # 'laus125'
    # dip_fname = op.join(MEG_DIR, subject, 'run3_Ictal.dip') # _ictal
    dip_fname = '/autofs/space/frieda_003/users/valia/epilepsy_clin/6966926_1426/200618/1426_EPI_lang.dip'
    # dip_fname = op.join(MEG_DIR, subject, 'EPI.dip')  # _ictal
    n_jobs = utils.get_n_jobs(20)
    n_jobs = n_jobs if n_jobs > 0 else 4
    print('jobs: {}'.format(n_jobs))
    #plot_dipole(dip_fname, subject)
    dipoles = parse_dip_file(dip_fname)
    mri_dipoles = convert_dipoles_to_mri_space(subject, dipoles, overwrite=True)
    # calc_distances_from_rois(subject)
    calc_dipoles_rois(subject, atlas=atlas, overwrite=True, n_jobs=n_jobs)