from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
import glob
import os.path as op

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


def calc_eeg_induced_power(subject, windows_fnames, baseline_name, inverse_method='dSPM', check_for_labels_files=True):
    bands = ['theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    non_zvlas_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'eeg', 'non-zvlas'))
    for window_fname in windows_fnames:
        fol = op.join(EEG_DIR, subject, '{}-epilepsy-{}-eeg-{}-induced_power'.format(
            subject, inverse_method, utils.namebase(window_fname)))
        if check_for_labels_files or not op.isdir(fol):
            args = eeg.read_cmd_args(dict(
                subject=subject,
                mri_subject=subject,
                function='calc_stc',
                task='epilepsy',
                calc_source_band_induced_power=True,
                evo_fname=window_fname,
                n_jobs=1,
                overwrite_stc=False
            ))
            eeg.call_main(args)
        for band in bands:
            stc_template = '{}-epilepsy-{}-eeg-{}_{}'.format(subject, inverse_method, '{window}', band)
            window_stc_name = stc_template.format(window=utils.namebase(window_fname))
            # ugly_fix(subject, window_stc_name, band, 'eeg')
            args = eeg.read_cmd_args(dict(
                subject=subject,
                mri_subject=subject,
                function='calc_stc_zvals',
                task='epilepsy',
                stc_name=window_stc_name,
                baseline_stc_name=stc_template.format(window=baseline_name),
                use_abs=1,
                overwrite_stc=False
            ))
            eeg.call_main(args)
    # stc_files = [f for f in glob.glob(op.join(MMVT_DIR, subject, 'eeg', '*.stc')) if not '-zvals-' in utils.namebase(f)]
    # for stc_fname in stc_files:
    #     utils.move_file(stc_fname, non_zvlas_fol, overwrite=True)


def ugly_fix(subject, window_stc_name, band, modality):
    fname1 = op.join(MMVT_DIR, subject, modality, '{}-rh.stc'.format(window_stc_name))
    fname2 = op.join(MMVT_DIR, subject, modality, '{}_{}-rh.stc'.format(utils.namebase(window_stc_name), band))
    if not op.isfile(fname1) and op.isfile(fname2):
        utils.rename_files(fname2, fname1)


def calc_meg_induced_power(subject, windows_fnames, baseline_name, inverse_method='dSPM', check_for_labels_files=True):
    bands = ['theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    non_zvlas_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'meg', 'non-zvlas'))
    for window_fname in windows_fnames:
        fol = op.join(MEG_DIR, subject, '{}-epilepsy-{}-meg-{}-induced_power'.format(
            subject, inverse_method, utils.namebase(window_fname)))
        if check_for_labels_files or not op.isdir(fol):
            meg_args = meg.read_cmd_args(dict(
                subject=subject,
                mri_subject=subject,
                function='calc_stc',
                task='epilepsy',
                calc_source_band_induced_power=True,
                evo_fname=window_fname,
                fwd_usingEEG=False,
                n_jobs=1,
                overwrite_stc=False
            ))
            meg.call_main(meg_args)
        for band in bands:
            stc_template = '{}-epilepsy-{}-meg-{}_{}'.format(subject, inverse_method, '{window}', band)
            window_stc_name = stc_template.format(window=utils.namebase(window_fname))
            # ugly_fix(subject, window_stc_name, band, 'meg')
            args = meg.read_cmd_args(dict(
                subject=subject,
                mri_subject=subject,
                function='calc_stc_zvals',
                task='epilepsy',
                stc_name=window_stc_name,
                baseline_stc_name=stc_template.format(window=baseline_name),
                use_abs=1,
                overwrite_stc=False
            ))
            meg.call_main(args)
    # stc_files = [f for f in glob.glob(op.join(MMVT_DIR, subject, 'meg', '*.stc')) if not '-zvals-' in utils.namebase(f)]
    # for stc_fname in stc_files:
    #     utils.move_file(stc_fname, non_zvlas_fol, overwrite=True)


if __name__ == '__main__':
    windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    baseline_name = '37'
    inverse_method = 'dSPM'
    check_for_labels_files = True
    calc_eeg_induced_power('nmr00857', windows, baseline_name, inverse_method, check_for_labels_files)
    calc_meg_induced_power('nmr00857', windows, baseline_name, inverse_method, check_for_labels_files)