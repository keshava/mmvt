import os.path as op
from collections import defaultdict
from src.utils import utils
from src.preproc import meg
import mne

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def parse_dip_file(dip_fname):
    '''
    Parse dip file and return dict of events
    :param dip_fname:
    :return: dict of events
    '''
    events = defaultdict(list)
    # Should read the whole dip file in loop to find all the events
    event = [162023.8, 162023.8, 30.7, 63.7, 69.6, 89.84, 69.58, -50.12, 26.78, 77.3]
    events['run2_162'].append(event)
    return events


def convert_dipoles_to_mri_space(subject, dipoles, overwrite=False):
    '''
    :param dipole:
    :return:
    '''
    info_fname, info_exist = meg.get_info_fname('')
    info = utils.load(info_fname)
    output_fname = op.join(utils.make_dir(op.join(MMVT_DIR, subject, 'meg')), 'dipoles.pkl')
    if op.isfile(output_fname) and not overwrite:
        return True
    # If the trans file doesn't exist, you should calculate it using mne-python / MNE-analyzer
    trans_file = meg.find_trans_file()
    trans = mne.transforms.read_trans(trans_file)
    head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
    dev_head_t = info['dev_head_t']
    meg_trans = mne.transforms.combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri')
    mri_dipoles = defaultdict(list)
    for dipole_name, dipoles in dipoles.items():
        for dipole in dipoles:
            # begin end(ms)  X (mm)  Y (mm)  Z (mm)  Q(nAm) Qx(nAm) Qy(nAm) Qz(nAm)  g(%)
            begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            mri_pos = mne.transforms.apply_trans(trans, [[x, y, z]])[0]
            dir_xyz = mne.transforms.apply_trans(trans, [[qx, qy, qz]])[0]
            mri_dipoles[dipole_name].append([begin_t, end_t, *mri_pos, q, *dir_xyz, gf])
    utils.save(mri_dipoles, output_fname)
    return op.isfile(output_fname)


if __name__ == '__main__':
    subject = 'nmr01391'
    dip_fname = op.join(MEG_DIR, subject, 'epi.dip')
    dipoles = parse_dip_file(dip_fname)
    mri_dipoles = convert_dipoles_to_mri_space(subject, dipoles, overwrite=False)
