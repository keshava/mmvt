import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os.path as op
import numpy as np
from itertools import product

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


def plot_sensors_windows(subject, windows, condition, modality, bad_channels, do_plot=False):
    import matplotlib.pyplot as plt
    import mne
    if modality == 'meeg':
        return
    plt.figure()
    sensors_types = ['eeg'] if modality == 'eeg' else ['mag', 'grad']
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'sensors'))
    all_data = {}
    for sensors_type in sensors_types:
        for window_fname in windows:
            window = utils.namebase(window_fname)
            evoked = mne.read_evokeds(window_fname)[0]
            if modality == 'eeg':
                ev = evoked.pick_types(meg=False, eeg=True, exclude=bad_channels)
            else:
                ev = evoked.pick_types(meg=sensors_type, eeg=False, exclude=bad_channels)
            # best_channel = np.argmax(np.max(np.abs(ev.crop(-0.1, 0.1).data), axis=1))
            # plt.plot(ev.times, ev.data[best_channel], label=window)
            plt.plot(ev.times, ev.data.mean(0), label=window)
        # plt.legend(loc=0)
        plt.title('{} {}'.format(condition, sensors_type))
        if do_plot:
            plt.show()
        else:
            plt.savefig(op.join(figures_fol, '{}-{}.jpg'.format(condition, sensors_type)), dpi=300)
            plt.close()


def plot_average_sensors(subject, windows, condition, modality, bad_channels):
    import mne
    import matplotlib.pyplot as plt
    if modality == 'meeg':
        return
    evokes = []
    info = None
    title = '{}-{}-{}-windows'.format(subject, modality, condition)
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'average-sensors'))
    fig_fname = op.join(figures_fol, '{}.jpg'.format(title))
    meg_evoked_fname = op.join(root_dir, '{}.fif'.format(title))
    for window_fname in windows:
        evoked = mne.read_evokeds(window_fname)[0]
        evoked = evoked.pick_types(meg=modality == 'meg', eeg=modality == 'eeg', exclude=bad_channels)
        if info is None:
            info = evoked.info
        evokes.append(evoked.data)
    evokes = np.array(evokes).mean(0)
    evoked_object = mne.EvokedArray(evokes, info, comment=title)
    fig = evoked_object.plot(window_title=title, spatial_colors=True, show=False)
    fig.tight_layout()
    plt.savefig(fig_fname, dpi=300)
    plt.close()
    mne.write_evokeds(meg_evoked_fname, evoked_object)


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
                      use_zvals=False, cond_name='interictals', stc_subfolder='zvals', stc_name='',
                      stc_downsample=2, con_threshold=0.5, bands=None):
    if bands is None:
        bands = utils.calc_bands(1, high_freq, include_all_freqs=True)
    if func_rois_atlas:
        con_indentifer = 'func_rois'
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'connectivity'))
    for band_name in bands.keys():
        d_cond, d_baseline, x_cond, x_baseline, names, stc_data, stc_times = calc_cond_and_basline(
            subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals, node_name,
            cond_name=cond_name, stc_subfolder=stc_subfolder, stc_name=stc_name, stc_downsample=stc_downsample)
        if x_cond is not None and x_baseline is not None:
            plot_norm_data(
                x_cond, x_baseline, names, condition, con_threshold, node_name, stc_data, stc_times,
                figures_fol=figures_fol)


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


def plot_norm_data(x_cond, x_baseline, con_names, condition, threshold, node_name, stc_data, stc_times, windows_len=100,
                   windows_shift=10, figures_fol='', ax=None):
    # con_norm = x_cond - x_baseline
    # con_norm = x_cond - x_cond[:, :200].mean(axis=1, keepdims=True)
    # baseline_std = np.std(x_baseline, axis=1, keepdims=True)
    # baseline_mean = np.mean(x_baseline, axis=1, keepdims=True)
    baseline_mean = np.max(x_cond[:, :200], axis=1, keepdims=True)
    baseline_std = np.std(x_cond[:, :200], axis=1, keepdims=True)

    con_norm = (x_cond - baseline_mean) / baseline_std
    fig_fname = op.join(figures_fol, 'ictal-baseline', '{}-connectivity-ictal-baseline.jpg'.format(condition))

    norm = {}
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    conn_conditions = list(product(['within', 'between'], utils.HEMIS))
    colors = ['c', 'b', 'k', 'm']
    lines, labels = [], []
    no_ord_con_names = [con_name.split(' ')[0] for con_name in con_names]
    for conn_type, color in zip(conn_conditions, colors):
        mask = epi_utils.filter_connections(node_name, con_norm, no_ord_con_names, threshold, conn_type, use_abs=False)
        if sum(mask) == 0:
            print('{} no connections {}'.format(condition, conn_type))
            continue
        else:
            print('{}: {} connection for {} {}'.format(condition, sum(mask), conn_type[0], conn_type[1]))
        names = np.array(con_names)[mask]
        norm[conn_type] = con_norm[mask]
        windows_num = norm[conn_type].shape[1]
        dt = (stc_times[-1] - stc_times[windows_len]) / windows_num
        # print('windows num: {} windows length: {:.2f}ms windows shift: {:2f}ms'.format(
        #     windows_num, (stc_times[windows_len] - stc_times[0]) * 1000, dt * 1000))
        time = np.arange(stc_times[windows_len], stc_times[-1], dt)[:-1]
        marker = '+' if conn_type[0] == 'within' else 'x'
        l = ax.scatter(time, norm[conn_type].max(0), color=color)#, marker=marker)
        lines.append(l)
        conn_type = (conn_type[0], 'right') if conn_type[1] == 'rh' else (conn_type[0], 'left')
        labels.append(' '.join(conn_type) if conn_type[0] == 'within' else '{} to {}'.format(*conn_type))

    if stc_data is not None:
        ax2 = ax.twinx()
        l = ax2.plot(stc_times[windows_len:], stc_data[windows_len:].T, 'y--', alpha=0.2) # stc_data[:-100].T
        lines.append(l[0])
        labels.append('Source normalized activity')
        # ax2.set_ylim([0.5, 4.5])
        # ax2.set_xlim([])
        # ax2.set_yticks(range(1, 5))
        ax2.set_ylabel('Source z-values', fontsize=12)
    # ax.set_xticks(time)
    # xticklabels = ['{}-{}'.format(t, t + windows_shift) for t in time]
    # xticklabels[2] = '{}\nonset'.format(xticklabels[2])
    # ax.set_xticklabels(xticklabels, rotation=30)
    ax.set_ylabel('Causality: Interictals\n minus Baseline', fontsize=12)
    # ax.set_yticks([0, 0.5])
    ax.set_ylim(bottom=0)#, 0.7])
    # ax.axvline(x=x_axis[10], color='r', linestyle='--')
    plt.title('{} ictal-baseline ({} connections)'.format(condition, x_cond.shape[0]))

    # labs = [*conn_conditions, 'Source normalized activity']
    # ax.legend([l1[conn_conditions[k]][0] for k in range(4)] + l2, labs, loc=0)
    # ax.legend([l1[conn_conditions[0]]] + [l1[conn_conditions[1]]] + l2, labs, loc=0)
    ax.legend(lines, labels, loc='upper right')#loc=0)
    plt.axvline(x=0, linestyle='--', color='k')
    # if ax is None:
    if figures_fol != '':
        plt.savefig(fig_fname, dpi=300)
        print('Figure was saved in {}'.format(fig_fname))
        plt.close()
    else:
        plt.show()


def get_cond_and_baseline_fnames(subject, con_method, modality, condition, extract_mode, band_name, con_indentifer,
                                 use_zvals, cond_name='interictals'):
    from src.preproc import connectivity
    template = connectivity.get_output_fname(
        subject, con_method, modality, extract_mode, '{}_{}_{}'.format(band_name, '{}_{}'.format(
            condition, '{cond}'), con_indentifer))
    input_fname = '{}{}.npz'.format(template.format(cond=cond_name)[:-4], '_zvals' if use_zvals else '')
    baseline_fname = '{}{}.npz'.format(template.format(cond='baseline')[:-4], '_zvals' if use_zvals else '')
    return input_fname.replace('__', '_'), baseline_fname.replace('__', '_')


def calc_cond_and_basline(subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals,
                          node_name, use_abs=True, threshold=0.7, window_length=25, stc_downsample=2,
                          cond_name='interictals', stc_subfolder='zvals', stc_name=''):
    import mne
    from src.preproc import connectivity

    input_fname, baseline_fname = get_cond_and_baseline_fnames(
        subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals, cond_name)
    if not op.isfile(input_fname) or not op.isfile(baseline_fname):
        # print('Can\'t find {}'.format(input_fname))
        return None, None, None, None, None, None, None

    stcs_fol = op.join(MMVT_DIR, subject, 'meg', stc_subfolder)
    if stc_name == '':
        stc_name = '{}-epilepsy-dSPM-meg-{}-average-amplitude-zvals-rh.stc'.format(subject, condition)
    stc_fname = op.join(stcs_fol, stc_name)
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
    # baseline_values1 = epi_utils.set_new_ords(d_baseline['con_values'], best_ords1)
    # baseline_values2 = epi_utils.set_new_ords(d_baseline['con_values2'], best_ords2)
    baseline_values1 = connectivity.find_best_ord(d_baseline['con_values'], return_ords=False)
    baseline_values2 = connectivity.find_best_ord(d_baseline['con_values2'], return_ords=False)

    mask1 = epi_utils.filter_connections(node_name, con_values1, d_cond['con_names'], threshold, '', use_abs)
    mask2 = epi_utils.filter_connections(node_name, con_values2, d_cond['con_names2'], threshold, '', use_abs)
    names = np.concatenate((d_cond['con_names'][mask1], d_cond['con_names2'][mask2]))
    if len(names) == 0:
        print('{} no connections'.format(condition))
        return None, None, None, None, None, None, None

    x_cond = np.concatenate((con_values1[mask1], con_values2[mask2]))
    x_baseline = np.concatenate((baseline_values1[mask1], baseline_values2[mask2]))
    if best_ords1 is not None and best_ords2 is not None:
        best_ords = np.concatenate((best_ords1[mask1], best_ords2[mask2]))
        names = ['{} {}'.format(name, int(best_ord)) for name, best_ord in zip(names, best_ords)]
    return d_cond, d_baseline, x_cond, x_baseline, names, stc_data, times


def plot_both_conditions(subject, conditions, modality, high_freq=120, con_method='wpli2_debiased',
                      extract_mode='mean_flip', func_rois_atlas=True, node_name='occipital', # ''lateraloccipital'
                      use_zvals=False, threshold=0.7, windows_len=100, windows_shift=10):
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
                print('connectivity files are not exist for condition {} band {}!'.format(condition, band_name))
                print(input_fname, baseline_fname)
                break
        if not files_exist:
            continue
        f, (axs) = plt.subplots(2, sharex=True, sharey=False)
        for ax, condition in zip(axs, conditions):
            d_cond, d_baseline, x_cond, x_baseline, names, stc_data, stc_times = calc_cond_and_basline(
                subject, con_method, modality, condition, extract_mode, band_name, con_indentifer, use_zvals, node_name,
                use_abs=False, threshold=threshold, stc_downsample=1)
            plot_norm_data(d_cond, d_baseline, condition, 0.1, node_name, stc_data, stc_times,
                           windows_len, windows_shift, ax)
        axs[1].set_xlabel('Time (ms)', fontsize=12)
        fig_fname = op.join(figures_fol, 'connectivity_both_conds_{}_{}.jpg'.format(con_method, band_name))
        print('Saving connectivity figure to {}'.format(fig_fname))
        plt.savefig(fig_fname)
        # plt.show()
    print('Done!')