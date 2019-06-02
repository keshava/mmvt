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


def calc_induced_power(subject, windows_fnames, modality, inverse_method='dSPM', check_for_labels_files=True):
    for window_fname in windows_fnames:
        calc_induced_power_per_window(subject, window_fname, modality, inverse_method, check_for_labels_files)


def calc_induced_power_per_window(subject, window_fname, modality, inverse_method='dSPM', check_for_labels_files=True):
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


def calc_induced_power_zvals(subject, windows_fnames, baseline_name, modality, bands, inverse_method='dSPM', n_jobs=4):
    params = [(subject, modality, window_fname, baseline_name, bands, inverse_method)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_induced_power_zvals_parallel, params, n_jobs)


def _calc_induced_power_zvals_parallel(p):
    subject, modality, window_fname, baseline_name, bands, inverse_method = p
    module = eeg if modality == 'eeg' else meg
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
    non_zvlas_fol = utils.make_dir(op.join(modality_fol, 'non-zvals'))
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*.stc'))
                 if '-epilepsy-' in utils.namebase(f) and not '-zvals-' in utils.namebase(f)]
    for stc_fname in stc_files:
        utils.move_file(stc_fname, non_zvlas_fol, overwrite=True)


def plot_stcs_files(subject, modality, n_jobs=4):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*-lh.stc'))
                 if '-epilepsy-' in utils.namebase(f) and '-zvals-' in utils.namebase(f) and
                 '-{}-'.format(modality) in utils.namebase(f)]
    print('{} files for {}'.format(len(stc_files), modality))
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'all_stcs', modality))
    utils.run_parallel(_plot_stcs_files_parallel, [(stc_fname, figures_fol) for stc_fname in stc_files], n_jobs)


def _plot_stcs_files_parallel(p):
    stc_fname, figures_fol = p
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
        plt.savefig(fig_fname, dpi=300)
        plt.close()


def plot_baseline(subject, baseline_name):
    stc_fnames = glob.glob(
        op.join(MMVT_DIR, subject, 'meg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name))) + \
        glob.glob(op.join(MMVT_DIR, subject, 'eeg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name)))
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'baseline'))
    for stc_fname in stc_fnames:
        plot_stc_file(stc_fname, figures_fol)


def plot_windows(subject, windows, modality, bands, inverse_method):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    for window_fname in windows:
        window_name = utils.namebase(window_fname)
        figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'epilepsy-per_window-figures'))
        stc_name = '{}-epilepsy-{}-{}-{}'.format(subject, inverse_method, modality, window_name)
        fig_fname = op.join(figures_fol, '{}.jpg'.format(stc_name))
        if op.isfile(fig_fname):
            print('{} already exist'.format(fig_fname))
            continue
        plt.figure()
        all_found = True
        for band in bands:
            stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name, band))
            if not op.isfile(stc_band_fname):
                print('Can\'t find {}!'.format(stc_band_fname))
                all_found = False
                break
            stc = mne.read_source_estimate(stc_band_fname)
            data = np.max(stc.data, axis=0)
            plt.plot(data.T)
        if all_found:
            plt.title(window_name)
            plt.legend(bands)
            print('Saving {}'.format(window_name))
            plt.savefig(fig_fname, dpi=300)
            plt.close()


def plot_freqs(subject, windows, modality, bands, inverse_method, max_t=0, subfol=''):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    for band in bands:
        plt.figure()
        for window_fname in windows:
            window_name = utils.namebase(window_fname)
            figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'runs_frontal_temporal'))
            if subfol != '':
                figures_fol = utils.make_dir(op.join(figures_fol, subfol))
            fig_fname = op.join(figures_fol, '{}-{}.jpg'.format(modality, band))
            if op.isfile(fig_fname):
                print('{} already exist'.format(fig_fname))
                break
            stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name, band))
            if not op.isfile(stc_band_fname):
                print('Can\'t find {}!'.format(stc_band_fname))
                break
            stc = mne.read_source_estimate(stc_band_fname)
            data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
            plt.plot(data.T)
        plt.title('{} {}'.format(modality, band))
        plt.legend([utils.namebase(w) for w in windows])
        print('Saving {} {}'.format(modality, band))
        plt.savefig(fig_fname, dpi=300)
        plt.close()


def plot_activity_modalities(subject, windows, modalities, inverse_method, max_t=0, overwrite=False):
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'amplitude'))
    for window_fname in windows:
        window_name = utils.namebase(window_fname)
        fig_fname = op.join(figures_fol, '{}-amplitude.jpg'.format(window_name))
        if op.isfile(fig_fname) and not overwrite:
            print('{} already exist'.format(fig_fname))
            break
        plt.figure()
        for modality in modalities:
            modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
            stc_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name))
            if not op.isfile(stc_fname):
                print('Can\'t find {}!'.format(stc_fname))
                break
            stc = mne.read_source_estimate(stc_fname)
            data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
            plt.plot(data.T)
        else:
            plt.title('{} amplitude'.format(window_name))
            plt.legend(modalities)
            print('Saving {} amplitude'.format(window_name))
            plt.savefig(fig_fname, dpi=300)
            plt.close()


def plot_modalities(subject, windows, modalities, bands, inverse_method, max_t=0, n_jobs=4):
    from itertools import product
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'modalities'))
    params = [(subject, modalities, inverse_method, figures_fol, band, window_fname, max_t)
              for (band, window_fname) in product(bands, windows)]
    utils.run_parallel(_plot_modalities_parallel, params, n_jobs)


def _plot_modalities_parallel(p):
    subject, modalities, inverse_method, figures_fol, band, window_fname, max_t = p
    window_name = utils.namebase(window_fname)
    plt.figure()
    for modality in modalities:
        modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
        fig_fname = op.join(figures_fol, '{}-{}.jpg'.format(window_name, band))
        if op.isfile(fig_fname):
            print('{} already exist'.format(fig_fname))
            break
        stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
            subject, inverse_method, modality, window_name, band))
        if not op.isfile(stc_band_fname):
            print('Can\'t find {}!'.format(stc_band_fname))
            break
        stc = mne.read_source_estimate(stc_band_fname)
        data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
        plt.plot(data.T)
    else:
        plt.title('{} {}'.format(window_name, band))
        plt.legend(modalities)
        print('Saving {} {}'.format(window_name, band))
        plt.savefig(fig_fname, dpi=300)
        plt.close()


def fix_amplitude_fnames(subject, bands):
    stcs_files = glob.glob(op.join(MMVT_DIR, subject, 'meg', '{}-epilepsy-*-zvals-?h.stc'.format(subject))) + \
                 glob.glob(op.join(MMVT_DIR, subject, 'eeg', '{}-epilepsy-*-zvals-?h.stc'.format(subject)))
    for stc_fname in stcs_files:
        stc_name = utils.namebase(stc_fname)[len('{}-epilepsy-'.format(subject)):-len('-zvals-lh')]
        if stc_name.endswith('amplitude'):
            continue
        if not any([stc_name.endswith(band) for band in bands]):
            stc_name += '_amplitude'
            stc_end = '-zvals-lh.stc' if stc_fname.endswith('-zvals-lh.stc') else '-zvals-rh.stc'
            new_stc_fname = op.join(
                utils.get_parent_fol(stc_fname), '{}-epilepsy-{}{}'.format(subject, stc_name, stc_end))
            print('{} -> {}'.format(stc_fname, new_stc_fname))
            utils.rename_files(stc_fname, new_stc_fname)


def create_evokeds_links(subject, windows):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'evoked'))
    for window_fname in windows:
        new_window_fname = op.join(fol, utils.namebase_with_ext(window_fname))
        if op.isfile(new_window_fname) or op.islink(new_window_fname):
            continue
        utils.make_link(window_fname, new_window_fname)



if __name__ == '__main__':
    # subject = 'nmr00857'
    # windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    # windows += ['/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/43.9s.fif']
    # baseline_windows = ['/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/37.3_BGprSzs.fif']
    # temporal_windows = [w for w in windows if '_Ts' in utils.namebase(w)]
    # frontal_windows = [w for w in windows if '_Fs' in utils.namebase(w)]
    # baseline_name = '37.3_BGprSzs'

    subject = 'nmr01321'
    windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/4272326_01321/MMVT_epochs/run1*.fif')
    baseline_windows = ['/autofs/space/frieda_001/users/valia/epilepsy/4272326_01321/MMVT_epochs/run1_Base_line__27.7s.fif']
    windows.remove(baseline_windows[0])

    windows_with_baseline = windows + baseline_windows
    baseline_name = utils.namebase(baseline_windows[0])
    inverse_method = 'dSPM'
    check_for_labels_files = False
    max_t = 7500
    modalities = ['eeg', 'meg', 'meeg']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    n_jobs = utils.get_n_jobs(-4)
    for modality in modalities:
        calc_induced_power(subject, baseline_windows, modality, inverse_method, check_for_labels_files)
        # calc_induced_power_zvals(subject, windows, baseline_name, modality, bands, inverse_method, n_jobs)
        # move_non_zvals_stcs(subject, modality)

        # plot_stcs_files(subject, modality, n_jobs)
        # plot_windows(subject, windows, modality, bands, inverse_method)
        # plot_freqs(subject, temporal_windows, modality, bands, inverse_method, max_t, 'temporal')
        # plot_freqs(subject, frontal_windows, modality, bands, inverse_method, max_t, 'frontal')

    # plot_modalities(subject, windows, modalities, bands, inverse_method, max_t, n_jobs)
    # plot_activity_modalities(subject, windows, modalities, inverse_method, overwrite=True)
    # plot_baseline(subject, baseline_name)
    # fix_amplitude_fnames(subject, bands)
    # create_evokeds_links(subject, windows_with_baseline)