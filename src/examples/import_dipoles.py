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


if __name__ == '__main__':
    subject = 'nmr01391'
    dip_fname = op.join(MEG_DIR, subject, 'run3_Ictal.dip') # _ictal
    n_jobs = utils.get_n_jobs(-10)
    n_jobs = n_jobs if n_jobs > 0 else 4
    print('jobs: {}'.format(n_jobs))
    #plot_dipole(dip_fname, subject)
    # dipoles = parse_dip_file(dip_fname)
    # mri_dipoles = convert_dipoles_to_mri_space(subject, dipoles, overwrite=True)
    calc_dipoles_rois(subject, atlas='laus125', overwrite=False, n_jobs=n_jobs)