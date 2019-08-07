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
    bands = utils.calc_bands(1, high_freq, include_all_freqs=True)
    if func_rois_atlas:
        con_indentifer = 'func_rois'
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'connectivity'))
    for band_name in bands.keys():
        d_cond, d_baseline, x_cond, x_baseline, names, stc_data = calc_cond_and_basline(
            subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals, node_name)
        if d_cond is None:
            continue
        x_axis = np.arange(x_cond.shape[1]) * 10
        plot_data(x_cond, x_baseline, x_axis, stc_data, condition, names)

        plot_norm_data(d_cond, d_baseline, x_axis, condition, 0.5, node_name, stc_data)
        # norm1_zvals = (d_cond['con_values'] - d_baseline['con_values'].mean(1, keepdims=True)) / \
        #               d_baseline['con_values'].std(1, keepdims=True)
        # norm2_zvals = (d_cond['con_values2'] - d_baseline['con_values2'].mean(1, keepdims=True)) / \
        #               d_baseline['con_values2'].std(1, keepdims=True)
        # plot_norm_data(norm1_zvals, norm2_zvals, 2)


def plot_data(x_cond, x_baseline, x_axis, stc_data, condition, names):
    import matplotlib.pyplot as plt
    time = np.arange(0, x_cond.shape[1], 5) * 10
    for x, cond in zip([x_cond, x_baseline], ['interictals', 'baseline']):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(x_axis, x.T)
        if stc_data is not None:
            ax2 = ax1.twinx()
            ax2.plot(stc_data.T, 'y--')
        ax1.set_xticks(time)
        ax1.set_xticklabels(['{}-{}'.format(t, t + 100) for t in time], rotation=30)
        # plt.axvline(x=x_axis[10], color='r', linestyle='--')
        # plt.axvline(x=x_axis[20], color='r', linestyle='--')
        plt.title('{}-{}'.format(condition, cond))
        plt.legend(names)
        plt.show()
        # plt.savefig(op.join(figures_fol, '{}-{}'.format(condition, cond)), dpi=300)
        # plt.close()


def plot_norm_data(d_cond, d_baseline, x_axis, condition, threshold, node_name, stc_data, stc_times, windows_len=25, windows_shift=10, ax=None):
    import matplotlib.pyplot as plt
    from src.preproc import connectivity
    # from src.mmvt_addon import colors_utils as cu

    norm1 = d_cond['con_values'] - d_baseline['con_values'].mean(1, keepdims=True)
    norm2 = d_cond['con_values2'] - d_baseline['con_values2'].mean(1, keepdims=True)
    norm1, best_ords1 = connectivity.find_best_ord(norm1, return_ords=True)
    norm2, best_ords2 = connectivity.find_best_ord(norm2, return_ords=True)
    norm = {}
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    from itertools import product
    conn_conditions = list(product(['within', 'between'], utils.HEMIS))

    # colors = cu.get_distinct_colors(4)
    colors = ['c', 'b', 'k', 'm']
    lines, labels = [], []
    # x_axis = x_axis [:-10]
    for conn_type, color in zip(conn_conditions, colors):
        mask1 = epi_utils.filter_connections(node_name, norm1, d_cond['con_names'], threshold, conn_type, use_abs=False)
        mask2 = epi_utils.filter_connections(node_name, norm2, d_cond['con_names2'], threshold, conn_type, use_abs=False)
        norm[conn_type] = np.concatenate((norm1[mask1], norm2[mask2]))#[:, :-10]
        names = np.concatenate((d_cond['con_names'][mask1], d_cond['con_names2'][mask2]))
        if best_ords1 is not None and best_ords2 is not None:
            best_ords = np.concatenate((best_ords1[mask1], best_ords2[mask2]))
            names = ['{} {}'.format(name, int(best_ord)) for name, best_ord in zip(names, best_ords)]
        if len(names) == 0 or max(norm[conn_type].max(0)) < 0:
            print('{} no connections {}'.format(condition, conn_type))
        else:
            windows_num = norm[conn_type].shape[1]
            dt = (stc_times[-1] - stc_times[windows_len]) / windows_num
            time = np.arange(stc_times[windows_len], stc_times[-1], dt)
            marker = '+' if conn_type[0] == 'within' else 'x'
            l = ax.scatter(time, norm[conn_type].max(0), color=color)#, marker=marker)
            lines.append(l)
            conn_type = (conn_type[0], 'right') if conn_type[1] == 'rh' else (conn_type[0], 'left')
            labels.append(' '.join(conn_type) if conn_type[0] == 'within' else '{} to {}'.format(*conn_type))

    if stc_data is not None:
        ax2 = ax.twinx()
        l = ax2.plot(stc_times[windows_len:], stc_data[windows_len:].T, 'y--') # stc_data[:-100].T
        lines.append(l[0])
        labels.append('Source normalized activity')
        ax2.set_ylim([0.5, 4.5])
        # ax2.set_xlim([])
        ax2.set_yticks(range(1, 5))
        ax2.set_ylabel('Source z-values', fontsize=12)
    # ax.set_xticks(time)
    # xticklabels = ['{}-{}'.format(t, t + windows_shift) for t in time]
    # xticklabels[2] = '{}\nonset'.format(xticklabels[2])
    # ax.set_xticklabels(xticklabels, rotation=30)
    ax.set_ylabel('Causality: Interictals\n minus Baseline', fontsize=12)
    # ax.set_yticks([0, 0.5])
    ax.set_ylim([0, 0.7])
    # ax.axvline(x=x_axis[10], color='r', linestyle='--')
    plt.title('{} interictals cluster'.format('Right' if condition == 'R' else 'Left'))

    # labs = [*conn_conditions, 'Source normalized activity']
    # ax.legend([l1[conn_conditions[k]][0] for k in range(4)] + l2, labs, loc=0)
    # ax.legend([l1[conn_conditions[0]]] + [l1[conn_conditions[1]]] + l2, labs, loc=0)
    ax.legend(lines, labels, loc=0)
    if ax is None:
        plt.show()
    # plt.savefig(op.join(figures_fol, '{} interictals-basline'.format(condition)), dpi=300)
    # plt.close()


def get_cond_and_baseline_fnames(subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals):
    from src.preproc import connectivity
    template = connectivity.get_output_fname(
        subject, con_method, modality, extract_mode, '{}_{}_{}'.format(band_name, '{}_{}'.format(
            condition, '{cond}'), con_indentifer))
    input_fname = '{}{}.npz'.format(template.format(cond='interictals')[:-4], '_zvals' if use_zvals else '')
    baseline_fname = '{}{}.npz'.format(template.format(cond='baseline')[:-4], '_zvals' if use_zvals else '')
    return input_fname, baseline_fname


def calc_cond_and_basline(subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals,
                          node_name, use_abs=True, threshold=0.7, window_length=25, stc_downsample=2):
    import mne
    from src.preproc import connectivity

    input_fname, baseline_fname = get_cond_and_baseline_fnames(
        subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals)
    if not op.isfile(input_fname) or not op.isfile(baseline_fname):
        # print('Can\'t find {}'.format(input_fname))
        return None, None, None, None, None, None

    stcs_fol = op.join(MMVT_DIR, subject, 'meg', 'zvals')
    stc_fname = op.join(
        stcs_fol, '{}-epilepsy-dSPM-meg-{}-average-amplitude-zvals-rh.stc'.format(subject, condition))
    if op.isfile(stc_fname):
        stc = mne.read_source_estimate(stc_fname)
        times = utils.downsample(stc.times, stc_downsample)# [window_length:]
        stc_data = np.max(stc.data, axis=0)
        stc_data = utils.downsample(stc_data, stc_downsample)# [window_length:]
    else:
        stc_data, times = None, None

    d_cond, d_baseline = np.load(input_fname), np.load(baseline_fname)
    con_values1, best_ords1 = connectivity.find_best_ord(d_cond['con_values'], return_ords=True)
    con_values2, best_ords2 = connectivity.find_best_ord(d_cond['con_values2'], return_ords=True)
    baseline_values1 = epi_utils.set_new_ords(d_baseline['con_values'], best_ords1)
    baseline_values2 = epi_utils.set_new_ords(d_baseline['con_values2'], best_ords2)

    mask1 = epi_utils.filter_connections(node_name, con_values1, d_cond['con_names'], threshold, '', use_abs)
    mask2 = epi_utils.filter_connections(node_name, con_values2, d_cond['con_names2'], threshold, '', use_abs)
    names = np.concatenate((d_cond['con_names'][mask1], d_cond['con_names2'][mask2]))
    if len(names) == 0:
        print('{} no connections'.format(condition))
        return None, None, None, None, None, None

    x_cond = np.concatenate((con_values1[mask1], con_values2[mask2]))
    x_baseline = np.concatenate((baseline_values1[mask1], baseline_values2[mask2]))
    if best_ords1 is not None and best_ords2 is not None:
        best_ords = np.concatenate((best_ords1[mask1], best_ords2[mask2]))
        names = ['{} {}'.format(name, int(best_ord)) for name, best_ord in zip(names, best_ords)]
    return d_cond, d_baseline, x_cond, x_baseline, names, stc_data, times


def plot_both_conditions(subject, conditions, modality, high_freq=120, con_method='wpli2_debiased',
                      extract_mode='mean_flip', func_rois_atlas=True, node_name='occipital', # ''lateraloccipital'
                      use_zvals=False, threshold=0.7, windows_len=25, windows_shift=10):
    import matplotlib.pyplot as plt

    bands = utils.calc_bands(1, high_freq, include_all_freqs=True)
    if func_rois_atlas:
        con_indentifer = 'func_rois'
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'connectivity'))
    for band_name in bands.keys():
        files_exist = True
        for condition in conditions:
            input_fname, baseline_fname = get_cond_and_baseline_fnames(
                subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals)
            files_exist = files_exist and op.isfile(input_fname) and op.isfile(baseline_fname)
        if not files_exist:
            continue
        f, (axs) = plt.subplots(2, sharex=True, sharey=False)
        for ax, condition in zip(axs, conditions):
            d_cond, d_baseline, x_cond, x_baseline, names, stc_data, stc_times = calc_cond_and_basline(
                subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals, node_name,
                use_abs=False, threshold=threshold, stc_downsample=1)
            x_axis = np.arange(x_cond.shape[1]) * 10
            plot_norm_data(d_cond, d_baseline, x_axis, condition, 0.1, node_name, stc_data, stc_times, windows_len, windows_shift, ax)
        axs[1].set_xlabel('Time (ms)', fontsize=12)
        plt.show()
    print('Done!')