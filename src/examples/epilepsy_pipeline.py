from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
import glob
import os.path as op
import mne
import numpy as np

import matplotlib.pyplot as plt
# plt.switch_backend('agg')


LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


def calc_induced_power_pipeline(
        subject, windows_fnames, baseline_name, modality, inverse_method='dSPM', check_for_labels_files=True):
    for window_fname in windows_fnames:
        calc_induced_power(subject, window_fname, modality, inverse_method, check_for_labels_files)
        calc_induced_power_zvals(subject, modality, window_fname, baseline_name, inverse_method)
    move_non_zvals_stcs(subject, modality)


def calc_induced_power(subject, window_fname, modality, inverse_method='dSPM', check_for_labels_files=True):
    root_dir = EEG_DIR if modality == 'eeg' else MEG_DIR
    module = eeg if modality == 'eeg' else meg
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


def calc_induced_power_zvals(subject, modality, window_fname, baseline_name, inverse_method):
    module = eeg if modality == 'eeg' else meg
    bands = ['theta', 'alpha', 'beta', 'gamma', 'high_gamma']
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


def move_non_zvals_stcs(subject, modality):
    # Move not zvals stc files
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    non_zvlas_fol = utils.make_dir(op.join(modality_fol, 'non-zvlas'))
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*.stc'))
                 if '-epilepsy-' in utils.namebase(f) and not '-zvals-' in utils.namebase(f)]
    for stc_fname in stc_files:
        utils.move_file(stc_fname, non_zvlas_fol, overwrite=True)


def plot_stcs_files(subject, modality):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*-lh.stc'))
                 if '-epilepsy-' in utils.namebase(f) and '-zvals-' in utils.namebase(f)]
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures'))
    for stc_fname in stc_files:
        plot_stc_file(stc_fname, figures_fol)


def plot_stc_file(stc_fname, figures_fol):
    stc_name = utils.namebase(stc_fname)[:-3]
    fig_fname = op.join(figures_fol, '{}.jpg'.format(stc_name))
    if not op.isfile(fig_fname):
        stc = mne.read_source_estimate(stc_fname)
        data = np.max(stc.data, axis=0)
        plt.figure()
        plt.plot(data.T)
        plt.title(utils.namebase(stc_fname)[:-3])
        print('Saving {}'.format(fig_fname))
        plt.savefig(fig_fname)
        plt.close()


def plot_baseline(subject, baseline_name):
    stc_fnames = glob.glob(op.join(MMVT_DIR, subject, 'meg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(
        subject, baseline_name)))
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-baseline-figures'))
    for stc_fname in stc_fnames:
        plot_stc_file(stc_fname, figures_fol)


def plot_windows(subject, windows, modality, inverse_method):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    bands = ['theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    for window_fname in windows:
        window_name = utils.namebase(window_fname)
        figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-per_window-figures'))
        stc_name = '{}-epilepsy-{}-{}-{}'.format(subject, inverse_method, modality, window_name)
        fig_fname = op.join(figures_fol, '{}.jpg'.format(stc_name))
        if op.isfile(fig_fname):
            continue
        plt.figure()
        for band in bands:
            stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name, band))
            stc = mne.read_source_estimate(stc_band_fname)
            data = np.max(stc.data, axis=0)
            plt.plot(data.T)
        plt.title(window_name)
        plt.legend(bands)
        print('Saving {}'.format(window_name))
        plt.savefig(fig_fname)
        plt.close()


if __name__ == '__main__':
    subject = 'nmr00857'
    windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    windows += ['/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/43.9s.fif',
        '/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/37.3_BGprSzs.fif']

    baseline_name = '37'
    inverse_method = 'dSPM'
    check_for_labels_files = False
    modalities = ['eeg', 'meg', 'meeg']
    for modality in modalities:
    #     calc_induced_power_pipeline(
    #         subject, windows, baseline_name, modality, inverse_method, check_for_labels_files)
        # plot_stcs_files(subject, modality)
        plot_windows(subject, windows, modality, inverse_method)
    # plot_baseline(subject, baseline_name)
