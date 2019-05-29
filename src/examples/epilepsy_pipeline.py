from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
import glob
import os.path as op
import mne
import numpy as np

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


def calc_induced_power(subject, windows_fnames, baseline_name, modality, inverse_method='dSPM',
                       check_for_labels_files=True):
    bands = ['theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    root_dir = EEG_DIR if modality == 'eeg' else MEG_DIR
    module = eeg if modality == 'eeg' else meg
    for window_fname in windows_fnames:
        fol = op.join(root_dir, subject, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, utils.namebase(window_fname)))
        if check_for_labels_files or not op.isdir(fol):
            args = module.read_cmd_args(dict(
                subject=subject,
                mri_subject=subject,
                function='calc_stc',
                task='epilepsy',
                calc_source_band_induced_power=True,
                fwd_usingEEG=modality in ['eeg', 'meeg'],
                evo_fname=window_fname,
                n_jobs=1,
                overwrite_stc=False
            ))
            module.call_main(args)
        for band in bands:
            stc_template = '{}-epilepsy-{}-{}-{}_{}'.format(subject, inverse_method, modality, '{window}', band)
            window_stc_name = stc_template.format(window=utils.namebase(window_fname))
            args = module.read_cmd_args(dict(
                subject=subject,
                mri_subject=subject,
                function='calc_stc_zvals',
                task='epilepsy',
                stc_name=window_stc_name,
                baseline_stc_name=stc_template.format(window=baseline_name),
                use_abs=1,
                overwrite_stc=False
            ))
            module.call_main(args)
    # Move not zvals stc files
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    non_zvlas_fol = utils.make_dir(op.join(modality_fol, 'non-zvlas'))
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*.stc'))
                 if '-epilepsy-' in utils.namebase(f) and not '-zvals-' in utils.namebase(f)]
    for stc_fname in stc_files:
        utils.move_file(stc_fname, non_zvlas_fol, overwrite=True)


def plot_stcs_files(subject, modality):
    import matplotlib.pyplot as plt
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*-lh.stc'))
                 if '-epilepsy-' in utils.namebase(f) and '-zvals-' in utils.namebase(f)]
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures'))
    for stc_fname in stc_files:
        stc_name = utils.namebase(stc_fname)[:-3]
        fig_fname = op.join(figures_fol, '{}.jpg'.format(stc_name))
        if op.isfile(fig_fname):
            continue
        stc = mne.read_source_estimate(stc_fname)
        data = np.max(stc.data, axis=0)
        plt.figure()
        plt.plot(data.T)
        plt.title(stc_name)
        print('Saving {}'.format(fig_fname))
        plt.savefig(fig_fname)
        plt.close()


if __name__ == '__main__':
    subject = 'nmr00857'
    windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    baseline_name = '37'
    inverse_method = 'dSPM'
    check_for_labels_files = False
    modalities = ['eeg', 'meg', 'meeg']
    for modality in ['meeg']:
        calc_induced_power(subject, windows, baseline_name, modality, inverse_method, check_for_labels_files)
        # plot_stcs_files(subject, modality)
