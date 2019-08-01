import matplotlib
matplotlib.use('TkAgg')

import os.path as op
import numpy as np
from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
from src.examples.epilepsy import utils as epi_utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def plot_topomaps(subject, modality, windows, bad_channels, parallel=True):
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'topomaps'))
    params = [(subject, modality, window_fname, bad_channels, figs_fol) for window_fname in windows]
    utils.run_parallel(_plot_topomaps_parallel, params, len(windows) if parallel else 1)


def _plot_topomaps_parallel(p):
    subject, modality, window_fname, bad_channels, figs_fol = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    evo_fname = op.join(MMVT_DIR, subject, 'evoked', '{}.fif'.format(window))
    if not op.isfile(evo_fname):
        utils.make_link(window_fname, evo_fname)
    fig_fname = op.join(figs_fol, '{}.jpg'.format(window))
    if op.isfile(fig_fname):
        return
    # if bad_channels != 'bads':
    #     bad_channels = bad_channels.split(',')
    module.plot_topomap(
        subject, evo_fname, times=[0], find_peaks=True, same_peaks=False, n_peaks=5, bad_channels=bad_channels,
        title=window, save_fig=True, fig_fname=fig_fname)


def plot_evokes(subject, modality, windows, bad_channels, parallel=True, overwrite=False):
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'evokes'))
    params = [(subject, modality, window_fname, bad_channels, figs_fol, overwrite) for window_fname in windows]
    utils.run_parallel(_plot_evokes_parallel, params, len(windows) if parallel else 1)


def _plot_evokes_parallel(p):
    subject, modality, window_fname, bad_channels, figs_fol, overwrite = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    evo_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'evoked'))
    evo_fname = op.join(evo_fol, '{}.fif'.format(window))
    if not op.isfile(evo_fname):
        utils.make_link(window_fname, evo_fname)
    fig_fname = op.join(figs_fol, '{}.jpg'.format(window))
    if op.isfile(fig_fname) and not overwrite:
        return
    # if bad_channels != 'bads':
    #     bad_channels = bad_channels.split(',')
    module.plot_evoked(
        subject, evo_fname, window_title=window, exclude=bad_channels, save_fig=True,
        fig_fname=fig_fname, overwrite=overwrite)


def plot_connectivity(subject, condition, modality, high_freq=120, con_method='wpli2_debiased',
                      extract_mode='mean_flip', func_rois_atlas=True, node_name='occipital', # ''lateraloccipital'
                      use_zvals=False):
    import matplotlib.pyplot as plt
    from src.preproc import connectivity
    import mne

    def plot_norm_data(norm1, norm2, threshold):
        norm1, best_ords1 = epi_utils.find_best_ord(norm1, return_ords=True)
        norm2, best_ords2 = epi_utils.find_best_ord(norm2, return_ords=True)
        mask1 = epi_utils.filter_connections(node_name, norm1, d_cond['con_names'], threshold)
        mask2 = epi_utils.filter_connections(node_name, norm2, d_cond['con_names2'], threshold)
        norm = np.concatenate((norm1[mask1], norm2[mask2]))
        best_ords = np.concatenate((best_ords1[mask1], best_ords2[mask2]))
        names = np.concatenate((d_cond['con_names'][mask1], d_cond['con_names2'][mask2]))
        names = ['{} {}'.format(name, int(best_ord)) for name, best_ord in zip(names, best_ords)]
        if len(names) == 0:
            print('{} {} no connections'.format(condition, cond))
            return

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        time = np.arange(0, norm.shape[1], 5) * 10
        plt.plot(x_axis, norm.T)
        if stc_data is not None:
            ax2 = ax1.twinx()
            ax2.plot(stc_data.T, 'y--')
        ax1.set_xticks(time)
        ax1.set_xticklabels(['{}-{}'.format(t, t + 100) for t in time], rotation=30)
        # plt.axvline(x=x_axis[10], color='r', linestyle='--')
        # plt.axvline(x=x_axis[20], color='r', linestyle='--')
        plt.title('{} interictals-basline'.format(condition))
        # plt.legend(names)
        plt.show()
        # plt.savefig(op.join(figures_fol, '{} interictals-basline'.format(condition)), dpi=300)
        # plt.close()

    bands = utils.calc_bands(1, high_freq, include_all_freqs=True)
    if func_rois_atlas:
        con_indentifer = 'func_rois'
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'connectivity'))
    stcs_fol = op.join(MMVT_DIR, subject, 'meg', 'zvals')
    for band_name in bands.keys():
        template = connectivity.get_output_fname(
            subject, con_method, modality, extract_mode, '{}_{}_{}'.format(band_name, '{}_{}'.format(
                condition, '{cond}'), con_indentifer))
        input_fname = '{}{}.npz'.format(template.format(cond='interictals')[:-4], '_zvals' if use_zvals else '')
        baseline_fname = '{}{}.npz'.format(template.format(cond='baseline')[:-4], '_zvals' if use_zvals else '')
        if not op.isfile(input_fname) or not op.isfile(baseline_fname):
            # print('Can\'t find {}'.format(input_fname))
            continue

        stc_fname = op.join(
            stcs_fol, 'nmr01327-epilepsy-dSPM-meg-{}-average-amplitude-zvals-rh.stc'.format(condition))
        if op.isfile(stc_fname):
            stc = mne.read_source_estimate(stc_fname)
            stc_data = np.max(stc.data, axis=0)
            stc_data = utils.downsample(stc_data, 2)[100:]
            # plt.figure()
            # plt.plot(stc_data.T)
        else:
            stc_data = None

        d_cond, d_baseline = np.load(input_fname), np.load(baseline_fname)
        con_values1, best_ords1 = epi_utils.find_best_ord(d_cond['con_values'], return_ords=True)
        con_values2, best_ords2 = epi_utils.find_best_ord(d_cond['con_values2'], return_ords=True)
        baseline_values1 = epi_utils.set_new_ords(d_baseline['con_values'], best_ords1)
        baseline_values2 = epi_utils.set_new_ords(d_baseline['con_values2'], best_ords2)

        mask1 = epi_utils.filter_connections(node_name, con_values1, d_cond['con_names'], 0.8)
        mask2 = epi_utils.filter_connections(node_name, con_values2, d_cond['con_names2'], 0.8)
        names = np.concatenate((d_cond['con_names'][mask1], d_cond['con_names2'][mask2]))
        if len(names) == 0:
            print('{} no connections'.format(condition))
            continue

        x_cond = np.concatenate((con_values1[mask1], con_values2[mask2]))
        x_baseline = np.concatenate((baseline_values1[mask1], baseline_values2[mask2]))
        best_ords = np.concatenate((best_ords1[mask1], best_ords2[mask2]))
        names = ['{} {}'.format(name, int(best_ord)) for name, best_ord in zip(names, best_ords)]
        time = np.arange(0, x_cond.shape[1], 5) * 10
        x_axis = np.arange(x_cond.shape[1]) * 10
        for x, cond in zip([x_cond, x_baseline], ['interictals', 'baseline']):
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            plt.plot(x_axis, x.T)
            if stc_data is not None:
                ax2 = ax1.twinx()
                ax2.plot(stc_data.T, 'y--')
            ax1.set_xticks(time)
            ax1.set_xticklabels(['{}-{}'.format(t, t+100) for t in time], rotation=30)
            # plt.axvline(x=x_axis[10], color='r', linestyle='--')
            # plt.axvline(x=x_axis[20], color='r', linestyle='--')
            plt.title('{}-{}'.format(condition, cond))
            # plt.legend(names)
            plt.show()
            # plt.savefig(op.join(figures_fol, '{}-{}'.format(condition, cond)), dpi=300)
            # plt.close()

        norm1_mean = d_cond['con_values'] - d_baseline['con_values'].mean(1, keepdims=True)
        norm2_mean = d_cond['con_values2'] - d_baseline['con_values2'].mean(1, keepdims=True)
        norm1_zvals = (d_cond['con_values'] - d_baseline['con_values'].mean(1, keepdims=True)) / \
                      d_baseline['con_values'].std(1, keepdims=True)
        norm2_zvals = (d_cond['con_values2'] - d_baseline['con_values2'].mean(1, keepdims=True)) / \
                      d_baseline['con_values2'].std(1, keepdims=True)
        plot_norm_data(norm1_mean, norm2_mean, 0.5)
        # plot_norm_data(norm1_zvals, norm2_zvals, 2)
