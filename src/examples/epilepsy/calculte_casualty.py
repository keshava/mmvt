import os.path as op
from src.utils import utils
from src.examples.epilepsy import pipeline
from glob import glob

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
ELECTRODES_DIR = utils.get_link_dir(LINKS_DIR, 'electrodes')


def find_remote_subject_dir(subject):
    remote_subjects_dir = '/space/megraid/clinical/MEG-MRI/seder/freesurfer'
    if not op.isdir(op.join(remote_subjects_dir, subject)):
        remote_subjects_dir = SUBJECTS_DIR
    remote_subject_dir = op.join(remote_subjects_dir, subject)
    if not op.isdir(remote_subject_dir):
        raise Exception('No reocon-all files!')
    # print(('No reocon-all files!'))
    return remote_subject_dir


def init_nmr01391():
    subject = 'nmr01391'
    remote_subject_dir = find_remote_subject_dir(subject)
    meg_fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/1391'
    if not op.isfile(meg_fol):
        meg_fol = op.join(MEG_DIR, subject)
    bad_channels = ['MEG{}'.format(c) for c in [
        '0113', '1532', '1623', '2042', '1912', '2032', '2522', '0642', '0121', '1421', '1221', '1023',
        '0741', '1022', '1242']]
    raw_fname = op.join(meg_fol, 'raw', '6859241_03_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, 'raw', '6859241_emptyroom_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def main(subject, modality, run_num, files_dict, inverse_method='MNE', overwrite_stc=False,
         overwrite_induced_power_zvals=False, n_jobs=4):
    from_index, to_index = None, None
    windows_with_baseline = files_dict['baseline'] + files_dict['ictal']
    pipeline.calc_amplitude(subject, modality, run_num, windows_with_baseline, inverse_method, overwrite_stc, n_jobs)
    # pipeline.calc_amplitude_zvals(
    #     subject, windows, baseline_name, modality, from_index, to_index, inverse_method,
    #     use_abs=False, parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)


if __name__ == '__main__':
    n_jobs = utils.get_n_jobs(-5)
    n_jobs = n_jobs if n_jobs > 1 else 1
    modality = 'meg'
    run_num = '3'
    fif_files, files_dict = [], {}

    subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01391()
    for subfol in ['baseline', 'ictal']:
        files = glob.glob(op.join(meg_fol, subfol, '*.fif'))
        fif_files += files
        files_dict[subfol] = files

    main(subject, modality, run_num, files_dict, inverse_method='MNE', overwrite_stc=False,
         overwrite_induced_power_zvals=False, n_jobs=4)