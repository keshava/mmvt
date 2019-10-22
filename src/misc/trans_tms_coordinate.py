import mne
from mne.transforms import apply_trans
import numpy as np
import os.path as op
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


def trans(trans_fname, tms_coordinate, output_fname):
    trans = mne.transforms.read_trans(trans_fname)
    head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
    sensors_pos = apply_trans(head_mri_t, np.array(tms_coordinate))
    # sensors_pos *= 1000
    print('Saving sensors pos in {}'.format(output_fname))
    print(sensors_pos)
    np.savez(output_fname, pos=sensors_pos, names=['tms'], picks=[])


if __name__ == '__main__':
    trans_fname = "/autofs/space/karima_003/users/consciousness/data/TMS/AR/20190731/AR-trans.fif"
    tms_coordinate = [24.64, 52.27, 75.87]
    output_fname = '/autofs/space/karima_001/users/elie/software/mmvt_root/mmvt_blend/AR/eeg/tms_sensors_positions.npz'
    trans(trans_fname, tms_coordinate, output_fname)