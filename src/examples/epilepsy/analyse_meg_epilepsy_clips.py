import os.path as op
import glob
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import os
import mne
from tqdm import tqdm

from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import meg, eeg
from src.preproc import connectivity
from src.preproc import electrodes

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
ELECTRODES_DIR = utils.get_link_dir(LINKS_DIR, 'electrodes')


def calc_anatomy(subject, atlas, remote_subject_dir, n_jobs):
    from src.preproc import  anatomy as anat
    args = anat.read_cmd_args(dict(
        subject=subject,
        atlas=atlas,
        function='all,check_bem',
        remote_subject_dir=remote_subject_dir,
        exclude='create_new_subject_blend_file',
        n_jobs=n_jobs
    ))
    anat.call_main(args)


def calc_fwd_inv(subject, raw_fname, bad_channels, empty_room_fname, remote_subject_dir, fwd_usingMEG, fwd_usingEEG,
                 overwrite, n_jobs):
    args = meg.read_cmd_args(dict(
        subject=subject,
        task='epilepsy',
        function='make_forward_solution,calc_inverse_operator',
        raw_fname=raw_fname,
        bad_channels=bad_channels,
        use_empty_room_for_noise_cov=True,
        empty_fname=empty_room_fname,
        fwd_usingMEG=fwd_usingMEG,
        fwd_usingEEG=fwd_usingEEG,
        remote_subject_dir=remote_subject_dir,
        overwrite_fwd=overwrite,
        overwrite_inv=overwrite,
        n_jobs=n_jobs
    ))
    meg.call_main(args)


def calc_labels_data(subject, fif_files, atlas, remote_subject_dir, bad_channels, fwd_usingMEG, fwd_usingEEG,
                     overwrite=False, n_jobs=4):
    labels_data_name = 'labels_data_epilepsy_{}_dSPM_mean_flip'.format(atlas)
    preproc_module = meg if fwd_usingMEG else eeg
    for fif_fname in fif_files:
        output_fname = op.join(
            MMVT_DIR, subject, 'meg', '{}_{}_{}.npz'.format(labels_data_name, utils.namebase(fif_fname), '{hemi}'))
        if utils.both_hemi_files_exist(output_fname) and not overwrite:
            print('lables data for {} already exist'.format(utils.namebase(fif_fname)))
            continue
        args = preproc_module.read_cmd_args(dict(
            subject=subject,
            task='epilepsy',
            function='calc_stc,calc_labels_avg_per_condition',
            atlas=atlas,
            evo_fname=fif_fname,
            bad_channels=bad_channels,
            fwd_usingMEG=fwd_usingMEG,
            fwd_usingEEG=fwd_usingEEG,
            modality=modality,
            stc_template=op.join(MMVT_DIR, subject, modality, '{}-epilepsy-{}-{}-{}'.format(
                subject, 'dSPM', modality, utils.namebase(fif_fname))),
            remote_subject_dir=remote_subject_dir,
            overwrite_stc=overwrite,
            overwrite_labels_data=overwrite,
            n_jobs=n_jobs
        ))
        preproc_module.call_main(args)


def calc_connectivity(subject, atlas, connectivity_method, files, bands, modality, overwrite=False, n_jobs=4):
    conn_fol = op.join(MMVT_DIR, subject, 'connectivity')
    now = time.time()
    for run, fif_fname in enumerate(files):
        utils.time_to_go(now, run, len(files), runs_num_to_print=1)
        file_name = utils.namebase(fif_fname)
        labels_data_name = 'labels_data_{}-epilepsy-{}-{}-{}_{}_mean_flip_{}.npz'.format(
            subject, 'dSPM', modality, file_name, atlas, '{hemi}')
        if not utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, modality, labels_data_name)):
            print('labels data does not exist for {}!'.format(file_name))
        for band_name, band_freqs in bands.items():
            output_fname = op.join(conn_fol, '{}_{}_{}_{}.npy'.format(modality, file_name, band_name, connectivity_method))
            if op.isfile(output_fname) and not overwrite:
                print('{} {} connectivity for {} already exist'.format(connectivity_method, band_name, file_name))
                continue
            print('calc_meg_connectivity: {}'.format(band_name))
            con_args = connectivity.read_cmd_args(utils.Bag(
                subject=subject,
                atlas=atlas,
                function='calc_lables_connectivity',
                connectivity_modality=modality,
                connectivity_method=connectivity_method,
                labels_data_name=labels_data_name,
                windows_length=500,
                windows_shift=100,
                identifier='{}_{}'.format(file_name, band_name),
                fmin=band_freqs[0],
                fmax=band_freqs[1],
                sfreq=2035, # clin MEG
                recalc_connectivity=False,
                save_mmvt_connectivity=True,
                threshold_percentile=80,
                n_jobs=n_jobs
            ))
            connectivity.call_main(con_args)


def analyze_graphs(subject, fif_files, graph_func, bands, modality, overwrite=False, n_jobs=4):
    now = time.time()
    for run, fif_fname in enumerate(fif_files):
        utils.time_to_go(now, run, len(fif_files), runs_num_to_print=1)
        for band_name in bands.keys():
            analyze_graph(subject, fif_fname, band_name, graph_func, modality, overwrite, n_jobs)


def analyze_graph(subject, fif_fname, band_name, graph_func, modality, overwrite=False, n_jobs=4):
    fol = op.join(MMVT_DIR, subject, 'connectivity')
    con_name = '{}_{}_{}_{}'.format(modality, utils.namebase(fif_fname), band_name, graph_func)
    output_fname = op.join(fol, '{}_mi.npy'.format(con_name))
    if op.isfile(output_fname) and not overwrite:
        print('{} already exists'.format(utils.namebase(output_fname)))
        return False
    input_fname = op.join(
        fol, '{}_{}_{}_mi.npy'.format(modality, utils.namebase(fif_fname), band_name))
    if not op.isfile(input_fname):
        print('{} does not exist!'.format(input_fname))
        return False
    print('Loading {}'.format(input_fname))
    con = np.load(input_fname).squeeze()
    values = analyze_graph_data(con, n_jobs)
    print('{}: min={}, max={}, mean={}'.format(con_name, np.min(values), np.max(values), np.mean(values)))
    print('Saving {}'.format(output_fname))
    np.save(output_fname, values)


def analyze_graph_data(con, n_jobs):
    T = con.shape[2]
    con[con < np.percentile(con, 80)] = 0
    indices = np.array_split(np.arange(T), n_jobs)
    chunks = [(con, indices_chunk, graph_func) for indices_chunk in indices]
    results = utils.run_parallel(_calc_graph_func, chunks, n_jobs)
    first = True
    for vals_chunk, times_chunk in results:
        if first:
            values = np.zeros((len(vals_chunk[0]), T))
            first = False
        values[:, times_chunk] = vals_chunk.T
    return values


def _calc_graph_func(p):
    con, times_chunk, graph_func = p
    vals = []
    now = time.time()
    for run, t in enumerate(times_chunk):
        utils.time_to_go(now, run, len(times_chunk), 10)
        con_t = con[:, :, t]
        g = nx.from_numpy_matrix(con_t)
        if graph_func == 'closeness_centrality':
            x = nx.closeness_centrality(g)
        elif graph_func == 'degree_centrality':
            x = nx.degree_centrality(g)
        elif graph_func == 'eigenvector_centrality':
            x = nx.eigenvector_centrality(g, max_iter=10000)
        elif graph_func == 'katz_centrality':
            x = nx.katz_centrality(g, max_iter=100000)
        else:
            raise Exception('Wrong graph func!')
        vals.append([x[k] for k in range(len(x))])
    vals = np.array(vals)
    return vals, times_chunk


def plot_graph_values(subject, file_name, con_name, func_name):
    output_fname = op.join(MMVT_DIR, subject, 'connectivity', file_name, '{}_{}.jpg'.format(con_name, func_name))
    if op.isfile(output_fname):
        return
    input_fname = op.join(MMVT_DIR, subject, 'connectivity', file_name, '{}_{}.npy'.format(con_name, func_name))
    if not op.isfile(input_fname):
        print('No {}!'.format(input_fname))
        return
    vals = np.load(input_fname)
    t_axis = np.linspace(-2, 5, vals.shape[1])
    plt.figure()
    plt.plot(t_axis, vals.T)
    plt.title('{} {}'.format(con_name, func_name))
    print('Saving figure to {}'.format(output_fname))
    plt.savefig(output_fname)
    plt.close()


def plot_all_files_graph_max(subject, baseline_fnames, event_fname, func_name, bands_names, modality, input_template,
                             sz_name='', sfreq=None, do_plot=False, overwrite=False):
    if sz_name == '':
        sz_name = utils.namebase(event_fname)
    output_fol = utils.make_dir(op.join(
        MMVT_DIR, subject, 'connectivity', '{}_{}'.format(modality, func_name), 'runs', sz_name))
    if modality == 'ieeg':
        # clip = np.load(event_fname)
        t_start, t_end = -5, 5
    else:
        clip = mne.read_evokeds(event_fname)[0]
        sfreq = clip.info['sfreq']
        t_start, t_end = clip.times[0], clip.times[-1]

    windows_length = 500
    half_window = (1 /sfreq) * (windows_length / 2) # In seconds
    scores = {}
    for band_name in bands_names:
        band_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity', '{}_{}'.format(modality, func_name), band_name))
        con_name = '{}_{}_mi'.format(modality, band_name)
        figure_name = '{}_{}_{}.jpg'.format(con_name, func_name, sz_name)
        output_fname = op.join(output_fol, figure_name)
        # if op.isfile(output_fname) and not overwrite:
        #     print('{} already exist'.format(figure_name))
        #     continue
        all_files_found = True
        for fname in baseline_fnames + [event_fname]:
            if input_template != '':
                file_name = utils.namebase(fname)
                input_fname = input_template.format(file_name=file_name, band_name=band_name)
                if not op.isfile(input_fname):
                    print('{} does not exist!!!'.format(input_fname))
                    all_files_found = False
                    break
        if not all_files_found:
            continue
        scores[band_name] = calc_score(
            event_fname, baseline_fnames, input_template, band_name, t_start, t_end, half_window,
            output_fname, band_fol, figure_name, do_plot)
    return scores

@utils.tryit()
def calc_score(event_fname, baseline_fnames, input_template, band_name, t_start, t_end, half_window,
               output_fname, band_fol, figure_name, do_plot):
    baseline_values = []
    if do_plot:
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111)
    for fname in baseline_fnames:
        val_fname = fname if input_template == '' else input_template.format(file_name=utils.namebase(fname))
        vals = np.load(val_fname)
        # max_ind = np.argmax(np.max(vals, axis=1))
        # max_vals = vals[max_ind]
        max_vals = np.max(vals, axis=0)
        # plt.plot(t_axis, np.argmax(vals, axis=0), 'g--')
        # plt.plot(t_axis, max_vals, 'g-', label=utils.namebase(fname))  # {}'.format(sz_ind))
        baseline_values.append(max_vals)
    baseline_values = np.array(baseline_values)

    event_fname = event_fname if input_template == '' else input_template.format(
        file_name=utils.namebase(event_fname), band_name=band_name)
    event_vals = np.load(event_fname)
    t_axis = np.linspace(t_start + half_window, t_end - half_window, event_vals.shape[1])
    # max_ind = np.argmax(np.max(event_vals, axis=1))
    # max_event_vals = event_vals[max_ind]
    # plt.plot(t_axis, np.argmax(event_vals, axis=0), 'b-')

    max_event_vals = np.max(event_vals, axis=0)
    max_t = np.argmax(max_event_vals)
    max_node = np.argmax(event_vals[:, max_t])
    if max_event_vals.shape > baseline_values[0].shape:
        max_event_vals = max_event_vals[:baseline_values[0].shape[0]]
    else:
        max_event_vals = np.pad(max_event_vals, (
            0, baseline_values[0].shape[0] - max_event_vals.shape[0]), 'constant', constant_values=np.nan)

    threshold_max = baseline_values.max(axis=0)
    threshold_ci = baseline_values.mean(axis=0) + 2 * baseline_values.std(axis=0)

    cross_indices = np.where((max_event_vals > threshold_ci) & (max_event_vals > threshold_max))
    score = sum(max_event_vals[cross_indices] - threshold_ci[cross_indices]) \
        if len(cross_indices) > 0 else 0

    if do_plot:
        plt.plot(t_axis, max_event_vals, 'b-', label='epi-event')  # {}'.format(sz_ind))
        plt.plot(t_axis, threshold_max, 'g--', label='baseline max')
        plt.plot(t_axis, threshold_ci, 'y--', label='baseline \mu+2\std')
        # plt.fill_between(t_axis, 0, threshold_ci,
        #                  color='#539caf', alpha=0.4, label='baseline min-max')
        plt.axvline(x=0, linestyle='--', color='k')
        plt.xlabel('Time(s)', fontsize=18)
        # xticklabels = np.arange(t_start, t_end).tolist()
        # xticklabels[3] = 'SZ'
        # axes.set_xticklabels(xticklabels)
        plt.ylabel('max(Eigencentrality)', fontsize=16)
        # plt.ylim([0.3, 0.6])
        plt.title('Eigencentrality ({})'.format(band_name), fontsize=16)
        plt.legend(loc='upper right', fontsize=16)

        plt.scatter(t_axis[cross_indices], max_event_vals[cross_indices], marker='x', c='r')
        print('Saving figure to {}'.format(utils.namebase(output_fname)))
        plt.savefig(op.join(band_fol, figure_name))
        # utils.copy_file(output_fname, op.join(band_fol, figure_name))
        plt.close()
    return score


def save_sz_pick_values(subject, files_names, func_name, atlas):
    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
    names = np.load(op.join(op.join(MMVT_DIR, subject, 'connectivity', 'labels_names.npy')))
    labels_indices = np.load(op.join(op.join(MMVT_DIR, subject, 'connectivity', 'labels_indices.npy')))
    names = names[labels_indices]
    for band_name in bands.keys():
        output_fname = op.join(output_fol, '{}_{}.npz'.format(func_name, band_name))
        # if op.isfile(output_fname):
        #     continue
        fname = [fname for fname in files_names if utils.namebase(fname).endswith('SZ')][0]
        file_name = utils.namebase(fname)
        con_name = 'meg_{}_mi'.format(band_name)
        input_fname = op.join(MMVT_DIR, subject, 'connectivity', file_name, '{}_{}.npy'.format(con_name, func_name))
        vals = np.load(input_fname)
        data_min, data_max = np.min(vals), np.max(vals)
        t_axis = np.linspace(-2, 5, vals.shape[1])
        print('onset index: {}'.format(np.where(t_axis > 0)[0][0]))
        # Find pick in time
        t_peak = np.argmax(np.max(vals, axis=0))
        max_vals = vals[:, t_peak]
        # all_vals = np.zeros((len(names)))
        # all_vals = [max_vals[list(names).index(l)] if l in names else 0 for l in labels]
        print('Saving {}'.format(output_fname))
        np.savez(output_fname, data=vals,
                 names=names, atlas=atlas, data_min=data_min, data_max=data_max,
                 title='{}-{}'.format(func_name, band_name), cmap='YlOrRd')
        for hemi in utils.HEMIS:
            labels_output_fname = op.join(MMVT_DIR, subject, 'meg', 'labels_data_epilepsy_laus125_{}_{}_{}.npz'.format(
                band_name, func_name, hemi))
            hemis = np.array([lu.get_hemi_from_name(l) == hemi for l in names])
            np.savez(labels_output_fname, data=vals[hemis], names=names[hemis], conditions=['sz'])


def plot_con(subject, files, band_name):
    fname = [fname for fname in files if utils.namebase(fname).endswith('SZ')][0]
    file_name = utils.namebase(fname)
    con_name = 'meg_{}_mi'.format(band_name)
    fol = op.join(MMVT_DIR, subject, 'connectivity', file_name)
    con = np.load(op.join(fol, '{}.npy'.format(con_name))).squeeze()
    print(np.unravel_index(np.argmax(con), con.shape))
    t_axis = np.linspace(-2, 5, con.shape[2])
    plt.plot(t_axis, con[0].T)
    plt.title(con_name)
    plt.show()


def calc_scores(subject, files_dict, graph_func, bands, modality, do_plot=False):
    scores = {}
    for event_name in ['ictal', 'IED', 'baseline']:
        scores[event_name] = {}
        for band_name in bands.keys():
            scores[event_name][band_name] = []

    input_template = op.join(
        MMVT_DIR, subject, 'connectivity', '{}_{}_{}_{}_mi.npy'.format(
            modality, '{file_name}', '{band_name}', graph_func))

    for event_name in ['ictal', 'IED']:
        for sz_fname in files_dict[event_name]:
            score = plot_all_files_graph_max(
                subject, files_dict['baseline'], sz_fname, graph_func, bands.keys(), modality, input_template,
                '', None, do_plot, overwrite=True)
            for band_name in bands.keys():
                if band_name in score:
                    scores[event_name][band_name].append(score[band_name])

    for sz_fname in files_dict['baseline']:
        new_baseline = utils.remote_items_from_list(files_dict['baseline'], [sz_fname])
        score = plot_all_files_graph_max(
            subject, new_baseline, sz_fname, graph_func, bands.keys(), modality, input_template,
            'baseline_{}'.format(utils.namebase(sz_fname)), None, do_plot, overwrite=True)
        if score is None:
            continue
        for band_name in bands.keys():
            if band_name in score:
                scores['baseline'][band_name].append(score[band_name])

    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity', '{}_{}'.format(modality, graph_func), 'scores'))
    names = ['ictal'] * len(files_dict['ictal']) + ['IED'] * len(files_dict['IED']) + \
            ['baseline'] * len(files_dict['baseline'])
    for f in files_dict['ictal'] + files_dict['IED'] + files_dict['baseline']:
        print(utils.namebase(f))
    for band_name in bands.keys():
        all_scores = scores['ictal'][band_name] + scores['IED'][band_name] + scores['baseline'][band_name]
        if len(band_name) == 0:
            print('No results for {}!'.format(band_name))
            continue
        x = range(len(all_scores))
        plt.bar(x, all_scores)
        plt.xticks(x, names, rotation=45)#'vertical')
        plt.savefig(op.join(output_fol, '{}_scores.jpg'.format(band_name)))
        plt.close()


def split_baseline(fif_fnames, clips_length=6, shift=6, overwrite=False):
    output_fol = op.join(utils.get_parent_fol(fif_fnames[0]), 'new_baselines')
    if not overwrite and op.isdir(output_fol) and len(glob.glob(op.join(output_fol, '*.fif'))) > 0:
        return glob.glob(op.join(output_fol, '*.fif'))
    utils.make_dir(output_fol)
    data = []
    for fif_fname in fif_fnames:
        clip = mne.read_evokeds(fif_fname)[0]
        freq = clip.info['sfreq']
        step = int(freq * clips_length)
        start_t, end_t = 0, len(clip.times)
        while start_t + clips_length * freq < end_t:
            new_clip = mne.EvokedArray(
                clip.data[:, start_t:start_t + step], clip.info, comment='baseline')
            data.append(new_clip.data[0, :10])
            mne.write_evokeds(op.join(output_fol, '{}_{}.fif'.format(
                utils.namebase(fif_fname), int(start_t / freq))), new_clip)
            start_t += int(freq * shift)
    data = np.array(data)
    return glob.glob(op.join(output_fol, '*.fif'))


def create_ictal_clips(subject, ictal_events_dict, ictal_template, overwrite=False, n_jobs=4):
    mmvt_root = op.join(MMVT_DIR, subject, 'electrodes')
    data_files, baseline_files = [], []
    meta = utils.Bag(np.load(op.join(mmvt_root, 'electrodes_meta_data.npz')))
    for ictal_id, times in ictal_events_dict.items():
        output_fname = op.join(mmvt_root, 'electrodes_data_{}.npy'.format(ictal_id))
        baseline_fname = op.join(mmvt_root, 'electrodes_baseline_{}.npy'.format(ictal_id))
        if op.isfile(output_fname) and op.isfile(baseline_fname) and not overwrite:
            data_files.append(output_fname)
            baseline_files.append(baseline_fname)
            continue
        event_fname = ictal_template.format(ictal_id=ictal_id)
        if not op.isfile(event_fname):
            print('Cannot find {}!'.format(event_fname))
            continue
        args = electrodes.read_cmd_args(utils.Bag(
            subject=subject,
            function='create_raw_data_from_edf',
            task='seizure',
            bipolar=False,
            raw_fname=event_fname,
            start_time=0,
            seizure_onset=times[0]-5, seizure_end=times[0]+5, # times[1],
            baseline_onset=0, baseline_end=100,
            time_format='seconds',
            lower_freq_filter=1,
            upper_freq_filter=150,
            power_line_notch_widths=5,
            remove_baseline=False,
            normalize_data = False,
            factor=1000,
            overwrite_raw_data=True,
            n_jobs=n_jobs
        ))
        electrodes.call_main(args)
        temp_output_fname = op.join(mmvt_root, 'electrodes_data_diff.npy')
        if op.isfile(temp_output_fname):
            os.rename(temp_output_fname, output_fname)
            data_files.append(output_fname)
        else:
            print('{}: no data!'.format(ictal_id))
        temp_baseline_fname = op.join(mmvt_root, 'electrodes_baseline.npy')
        if op.isfile(temp_baseline_fname):
            os.rename(temp_baseline_fname, baseline_fname)
            baseline_files.append(baseline_fname)
        else:
            print('{}: No baseline!'.format(ictal_id))
    meta_fname = op.join(mmvt_root, 'electrodes_meta_data_diff.npz')
    if op.isfile(meta_fname):
        os.rename(meta_fname, op.join(mmvt_root, 'electrodes_meta_data.npz'))
    return data_files, baseline_files


def get_ieeg_connectivity_files(subject, connectivity_method, graph_func, bands):
    mmvt_root = op.join(MMVT_DIR, subject, 'electrodes')
    input_fol = op.join(mmvt_root, '{}_{}'.format(connectivity_method, graph_func))
    data_files = glob.glob(op.join(mmvt_root, 'electrodes_data_*.npy'))
    ictal_ids = sorted(list(set([utils.namebase(f).split('_')[2] for f in data_files])))
    for id in ictal_ids:
        for band_name in bands.keys():
            data_fname = op.join(input_fol, 'electrodes_data_{}_{}_{}_{}.npy'.format(
                id, connectivity_method, graph_func, band_name))
            if not op.isfile(data_fname):
                continue
            baseline_fnames = glob.glob(op.join(input_fol, 'electrodes_baseline_{}_{}_{}_*_{}.npy'.format(
                id, connectivity_method, graph_func, band_name)))
            yield data_fname, baseline_fnames, band_name, id


def calc_ieeg_connectivity(subject, connectivity_method, graph_func, sfreq, big_window_length,
                           windows_length, windows_shift, overwrite=False, n_jobs=4):
    mmvt_root = op.join(MMVT_DIR, subject, 'electrodes')
    output_fol = utils.make_dir(op.join(mmvt_root, '{}_{}'.format(connectivity_method, graph_func)))
    files_dict = {}
    data_files = glob.glob(op.join(mmvt_root, 'electrodes_data_*.npy'))
    ictal_ids = sorted(list(set([utils.namebase(f).split('_')[2] for f in data_files])))
    now = time.time()
    for run, id in enumerate(ictal_ids):
        utils.time_to_go(now, run, len(ictal_ids), 1)
        data_fname = op.join(mmvt_root, 'electrodes_data_{}.npy'.format(id))
        files_dict[utils.namebase(data_fname)] = {'baseline': [], 'ictal': [data_fname]}
        data = np.load(data_fname).squeeze()
        for band_name, (fmin, fmax) in bands.items():
            output_fname = op.join(output_fol, '{}_{}_{}_{}.npy'.format(
                utils.namebase(data_fname), connectivity_method, graph_func, band_name))
            if op.isfile(output_fname) and not overwrite:
                continue
            print(utils.namebase(output_fname))
            con = connectivity.calc_mi(data, windows_length, windows_shift, sfreq, fmin, fmax, n_jobs)
            graph_data = analyze_graph_data(con, n_jobs)
            np.save(output_fname, graph_data)

        baseline_fname = op.join(mmvt_root, 'electrodes_baseline_{}.npy'.format(id))
        data = np.load(baseline_fname).squeeze()
        w = sfreq * big_window_length
        windows = connectivity.calc_windows(data.shape[1], w, w)
        for base_ind, w in enumerate(windows):
            for band_name, (fmin, fmax) in bands.items():
                output_fname = op.join(output_fol, '{}_{}_{}_{}_{}.npy'.format(
                    utils.namebase(baseline_fname), connectivity_method, graph_func, base_ind, band_name))
                files_dict[utils.namebase(data_fname)]['baseline'].append(output_fname)
                if op.isfile(output_fname) and not overwrite:
                    continue
                print(utils.namebase(output_fname))
                con = connectivity.calc_mi(data[:, w[0]:w[1]], windows_length, windows_shift, sfreq, fmin, fmax, n_jobs)
                graph_data = analyze_graph_data(con, n_jobs)
                np.save(output_fname, graph_data)
    return files_dict


def save_influmax_labels_values(subject, modality, atlas, event_name, connectivity_method, band, inverse_method='dSPM',
                                extract_mode='mean_flip'):
    file_name = '{}_{}_{}_{}_{}'.format(modality, event_name, band, graph_func, connectivity_method)
    con_fol = op.join(MMVT_DIR, subject, 'connectivity')
    event_vals = np.load(op.join(con_fol, '{}.npy'.format(file_name)))
    max_event_vals = np.max(event_vals, axis=0)
    max_t = np.argmax(max_event_vals)
    data = event_vals # [:, max_t]
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
    d = np.load(op.join(con_fol, '{}_{}_{}_{}.npz'.format(modality, event_name, band, connectivity_method)))
    labels_names = d['labels']
    np.savez(op.join(fol, '{}_mean.npz'.format(file_name)), names=labels_names,
             atlas=atlas, data=data, title='influmax',
             data_min=np.min(data), data_max=np.max(data), cmap='YlOrRd')
    labels_data_template = op.join(MMVT_DIR, subject, 'meg', 'labels_data_{}'.format(event_name))
    labels_data_template += '_{}_{}_{}_{}_{}.npz'  # task, atlas, inverse_method, em, hemi
    for hemi in utils.HEMIS:
        hemi_inds = [ind for (ind, label_name) in enumerate(labels_names) if label_name.endswith(hemi)]
        hemi_data = {extract_mode:data[hemi_inds]}
        meg.save_labels_data(
            hemi_data, hemi, labels_names[hemi_inds], atlas, ['epilepsy'], [extract_mode], [inverse_method],
            labels_data_template, task='epilepsy')
        meg_labels_fname = op.join(MMVT_DIR, subject, 'meg', 'labels_data_{}-epilepsy-{}-{}-{}_{}_{}_{}.npz'.format(
            subject, inverse_method, modality, event_name, atlas, extract_mode, hemi))
        meg_data = np.load(meg_labels_fname)
        label_name = 'middletemporal_4-rh'
        if label_name not in meg_data['names']:
            continue
        ind = list(meg_data['names']).index(label_name)
        label_data = meg_data['data'][ind]
        t_axis = np.linspace(-2, 5, label_data.shape[0])
        plt.plot(t_axis, label_data)
        plt.axvline(x=0, linestyle='--', color='k')
        plt.xlabel('Time(s)', fontsize=18)
        plt.ylabel(inverse_method, fontsize=18)
        plt.savefig(op.join(MMVT_DIR, subject, modality, '{}.jpg'.format(label_name)))
        plt.close()


def save_influmax_electrodes_values(subject, clip_ind, connectivity_method, graph_func, band):
    data = np.load(op.join(MMVT_DIR, subject, 'electrodes', '{}_{}'.format(connectivity_method, graph_func),
        'electrodes_data_{}_{}_{}_{}.npy'.format(clip_ind, connectivity_method, graph_func, band)))
    meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_meta_data.npz')
    meta = np.load(meta_fname)
    times = np.linspace(-5, 5, data.shape[0])
    conditions = ['seizure']
    np.savez(meta_fname, names=meta['names'], conditions=conditions, times=times)
    np.save(op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_data.npy'), data)


def init_nmr01391():
    subject = 'nmr01391'
    remote_subject_dir = find_remote_subject_dir(subject)
    meg_fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/1391' # '/homes/5/npeled/space1/MEG/nmr00857/clips'
    if not op.isfile(meg_fol):
        meg_fol = op.join(MEG_DIR, subject)
    bad_channels = ['MEG{}'.format(c) for c in [
        '0113', '1532', '1623', '2042', '1912', '2032', '2522', '0642', '0121', '1421', '1221', '1023',
        '0741', '1022', '1242']]
    raw_fname = op.join(meg_fol, 'raw', '6859241_03_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, 'raw', '6859241_emptyroom_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def init_nmr00857():
    subject = 'nmr00857' # 'nmr00857'
    remote_subject_dir = find_remote_subject_dir(subject)
    meg_fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI'
    bad_channels = ['MEG{}'.format(c) for c in [
        '0112', '0123', '0113', '2541', '2531', '1413', '2013', '0743', '0142']]
    bad_channels += ['EEG{}'.format(c) for c in [
        '016', '017', '026', '027', '041',  '042']]
    raw_fname = op.join(meg_fol, '5241495_01_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, '5241495_roomnoise_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def init_nmr01321():
    subject = 'nmr01321'
    remote_subject_dir = find_remote_subject_dir(subject)
    remote_meg_fol = '/autofs/space/violet_001/users/valia/epilepsy2019/4272326_01321/190501/'
    meg_fol = op.join(MEG_DIR, subject, 'clips')
    bad_channels = 'EEG001,EEG003,EEG004,EEG005,EEG008,EEG034,EEG045,EEG051,EEG057,EEG058,EEG060,EEG061,EEG062,EEG074,MEG1422,MEG1532,MEG2012,MEG2022'
    raw_fname = op.join(remote_meg_fol, '4272326_03_raw_ssst.fif')
    empty_room_fname = op.join(remote_meg_fol, '4272326_noiseroom_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def init_nmr01325():
    subject = 'nmr01325'
    remote_subject_dir = find_remote_subject_dir(subject)
    remote_meg_fol = '/cluster/neuromind/valia/epilepsy/6645962_01325/190523'
    meg_fol = op.join(MEG_DIR, subject, 'clips')
    bad_channels = 'EEG020,EEG021,EEG050,EEG051'
    raw_fname = op.join(remote_meg_fol, '6645962_01_raw_ssst.fif')
    empty_room_fname = op.join(remote_meg_fol, '6645962_noiseroom_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def init_mg112():
    subject = 'mg112'
    ictal_event = {
        1: [3622.29, 3668.28], 13: [1995.88, 2059.32], 14: [3585.09, 3636.38], 15: [3613.14, 3664.62],
        16: [3563.04,3613.03], 17: [10.05, 136.16], 18:	[2601.04, 2666.55], 19:	[6342.66, 6408.76],
        2: [3648.4, 3698.31], 3: [3569.75,	3626.26], 4: [2062.64, 2127.62], 5:	[3714.91, 3755.33],
        6: [3592.17, 3659.83], 7: [3624.54,	3672.23], 8: [139.75, 238.39]}
    ictal_template = op.join(ELECTRODES_DIR, subject, 'MG112_Seizure{ictal_id}.edf')
    return subject, ictal_event, ictal_template


def init_nmr01327():
    subject = 'nmr01327'
    evokes_fol = [d for d in [
        # '/autofs/space/frieda_001/users/valia/epilepsy/6600387_01327/epochs', #/right-left',
        op.join(MMVT_DIR, subject, 'evokes')] if op.isdir(d)][0]
    meg_fol = [d for d in [
        # '/autofs/space/frieda_001/users/valia/epilepsy/6600387_01327/190626',
        op.join(MEG_DIR, subject)] if op.isdir(d)][0]
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG059,EEG019,MEG1532'
    baseline_name = 'baseline_run1_195.7_12sec'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name, True


def find_remote_subject_dir(subject):
    remote_subjects_dir = '/space/megraid/clinical/MEG-MRI/seder/freesurfer'
    if not op.isdir(op.join(remote_subjects_dir, subject)):
        remote_subjects_dir = SUBJECTS_DIR
    remote_subject_dir = op.join(remote_subjects_dir, subject)
    # if not op.isdir(remote_subject_dir):
    #     raise Exception('No reocon-all files!')
    print(('No reocon-all files!'))
    return remote_subject_dir


def analyze_meeg(graph_func, connectivity_method, bands, overwrite, n_jobs):
    atlas = 'laus125'
    subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr00857()
    # subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01391()
    # subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01321()
    # subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01325()

    fwd_usingMEG, fwd_usingEEG = True, False
    modality = meg.get_modality(fwd_usingMEG, fwd_usingEEG)
    if not op.isdir(meg_fol):
        meg_fol = op.join(MEG_DIR, subject)
    if not op.isdir(meg_fol):
        print('No MEG fol!')
        return
    fif_files, files_dict = [], {}
    for subfol in ['baseline', 'ictal', 'IED']:
        files = glob.glob(op.join(meg_fol, subfol, '*.fif'))
        fif_files += files
        files_dict[subfol] = files

    # files_dict['baseline'] = split_baseline(files_dict['baseline'], clips_length=6, shift=6)
    # calc_anatomy(subject, atlas, remote_subject_dir, n_jobs)
    # calc_fwd_inv(
    #     subject, raw_fname, bad_channels, empty_room_fname, remote_subject_dir, fwd_usingMEG, fwd_usingEEG,
    #     overwrite, n_jobs)
    # calc_labels_data(
    #     subject, fif_files, atlas, remote_subject_dir, bad_channels, fwd_usingMEG, fwd_usingEEG, overwrite, n_jobs)
    bands = dict(gamma=[30, 55])
    # calc_connectivity(subject, atlas, connectivity_method, fif_files, bands, modality, True, n_jobs)
    # analyze_graphs(subject, fif_files, graph_func, bands, modality, overwrite, n_jobs)
    # calc_scores(subject, files_dict, graph_func, bands, modality, do_plot=True)
    save_influmax_labels_values(subject, modality, atlas, 'run1_45.6s_SZstart_MEG', connectivity_method, 'gamma')


def analyze_ieeg(graph_func, connectivity_method, windows_length, windows_shift, bands, overwrite, n_jobs):
    subject, ictal_event, ictal_template = init_mg112()
    sfreq, big_window_length = 512, 10
    # data_files, baseline_files = create_ictal_clips(subject, ictal_event, ictal_template, overwrite, n_jobs)
    # all_files_dict = calc_ieeg_connectivity(
    #     subject, connectivity_method, graph_func, sfreq, big_window_length,
    #     windows_length, windows_shift, overwrite, n_jobs)
    save_influmax_electrodes_values(subject, 5, connectivity_method, graph_func, 'gamma')
    return
    do_plot = True
    scores = {'ictal': {band: [] for band in bands.keys()}, 'baseline': {band: [] for band in bands.keys()}}
    for data_fname, baseline_fnames, band_name, ictal_id in get_ieeg_connectivity_files(
            subject, connectivity_method, graph_func, bands):
        score = plot_all_files_graph_max(
            subject, baseline_fnames, data_fname, graph_func, [band_name], 'ieeg', '', ictal_id, sfreq,
            do_plot=do_plot, overwrite=True)
        scores['ictal'][band_name].append(score[band_name])
        for ind, sz_fname in enumerate(baseline_fnames):
            new_baseline = utils.remote_items_from_list(baseline_fnames, [sz_fname])
            score = plot_all_files_graph_max(
                subject, new_baseline, sz_fname, graph_func, [band_name], 'ieeg', '',
                'baseline_{}_{}'.format(ictal_id, ind), sfreq, do_plot, overwrite=True)
            if score is not None:
                scores['baseline'][band_name].append(score[band_name])


if __name__ == '__main__':
    graph_func = 'eigenvector_centrality'
    connectivity_method = 'mi' # Mutual information
    bands = dict(all=[1, 120], delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    overwrite = False
    ieeg_windows_length, ieeg_windows_shift = int((1000/512) * 500),  int((1000/512) * 100)
    n_jobs = utils.get_n_jobs(-5)
    n_jobs = n_jobs if n_jobs > 1 else 1
    print('n_jobs = {}'.format(n_jobs))

    # analyze_meeg(graph_func, connectivity_method, bands, overwrite, n_jobs)
    analyze_ieeg(graph_func, connectivity_method, ieeg_windows_length, ieeg_windows_shift, bands, overwrite, n_jobs)
