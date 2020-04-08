import os.path as op
import glob
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import os

from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import meg, eeg
from src.preproc import connectivity

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


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
            function='calc_labels_avg_per_condition', # calc_stc
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
        # rename labels data file name
        # for hemi in utils.HEMIS:
        #     labels_data_fname = op.join(
        #         MMVT_DIR, subject, modality, '{}_{}.npz'.format(labels_data_name, hemi))
        #     if not op.isfile(labels_data_fname):
        #         raise Exception('No labels data! ({})'.format(labels_data_fname))
        #     new_lables_data_fname_hemi = output_fname.format(hemi=hemi)
        #     utils.delete_file(new_lables_data_fname_hemi)
        #     print('Renaming to {}'.format(new_lables_data_fname_hemi))
        #     os.rename(labels_data_fname, new_lables_data_fname_hemi)


def calc_connectivity(subject, atlas, connectivity_method, files, bands, modality, overwrite=False, n_jobs=4):
    conn_fol = op.join(MMVT_DIR, subject, 'connectivity')
    for fif_fname in files:
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
                recalc_connectivity=True,
                save_mmvt_connectivity=False,
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
    print('{}: min={}, max={}, mean={}'.format(con_name, np.min(values), np.max(values), np.mean(values)))
    print('Saving {}'.format(output_fname))
    np.save(output_fname, values)


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


def plot_all_files_graph_max(subject, baseline_fnames, event_fname, func_name, bands, sz_name='',
                             do_plot=False, overwrite=False):
    if sz_name == '':
        sz_name = utils.namebase(event_fname)
    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity', func_name, sz_name))
    sfreq = 2035
    windows_length = 500
    half_window = (1 /sfreq) * (windows_length / 2) # In seconds
    scores = {}
    for band_name in bands.keys():
        band_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity', func_name, band_name))
        con_name = 'meg_{}_mi'.format(band_name)
        figure_name = '{}_{}_{}.jpg'.format(con_name, func_name, sz_name)
        output_fname =  op.join(output_fol, figure_name)
        input_template = op.join(
            MMVT_DIR, subject, 'connectivity', 'meg_meg_{}_{}_{}_mi.npy'.format('{file_name}', band_name, func_name))
        # if op.isfile(output_fname) and not overwrite:
        #     print('{} already exist'.format(figure_name))
        #     continue
        all_files_found = True
        for fname in baseline_fnames + [event_fname]:
            file_name = utils.namebase(fname)
            if not op.isfile(input_template.format(file_name=file_name)):
                print('{} does not exist!!!'.format(input_template.format(file_name=file_name)))
                all_files_found = False
                break
        if not all_files_found:
            continue
        baseline_values = []
        all_values = []
        # sz_ind = 0
        if do_plot:
            fig = plt.figure(figsize=(10, 10))
            axes = fig.add_subplot(111)
        first = True
        for fname in baseline_fnames:
            vals = np.load(input_template.format(file_name=utils.namebase(fname)))
            if first:
                t_axis = np.linspace(-2 + half_window, 5 - half_window, vals.shape[1])
                first = False
            # max_ind = np.argmax(np.max(vals, axis=1))
            # max_vals = vals[max_ind]
            max_vals = np.max(vals, axis=0)
            # plt.plot(t_axis, np.argmax(vals, axis=0), 'g--')
            # plt.plot(t_axis, max_vals, 'g-', label=utils.namebase(fname))  # {}'.format(sz_ind))
            baseline_values.append(max_vals)
        baseline_values = np.array(baseline_values)

        event_vals = np.load(input_template.format(file_name=utils.namebase(event_fname)))
        # max_ind = np.argmax(np.max(event_vals, axis=1))
        # max_event_vals = event_vals[max_ind]
        # plt.plot(t_axis, np.argmax(event_vals, axis=0), 'b-')

        max_event_vals = np.max(event_vals, axis=0)
        max_event_vals = np.pad(max_event_vals, (0, baseline_values[0].shape[0] - max_event_vals.shape[0]), 'constant',
                          constant_values=np.nan)

        threshold_max = baseline_values.max(axis=0)
        threshold_ci = baseline_values.mean(axis=0) + 2 * baseline_values.std(axis=0)

        cross_indices = np.where((max_event_vals > threshold_ci) & (max_event_vals > threshold_max))
        scores[band_name] = sum(max_event_vals[cross_indices] - threshold_ci[cross_indices]) \
            if len(cross_indices) > 0 else 0

        if do_plot:
            plt.plot(t_axis, max_event_vals, 'b-', label='epi-event')  # {}'.format(sz_ind))
            plt.plot(t_axis, threshold_max, 'g--', label='baseline max')
            plt.plot(t_axis, threshold_ci, 'y--', label='baseline \mu+2\std')
            # plt.fill_between(t_axis, 0, threshold_ci,
            #                  color='#539caf', alpha=0.4, label='baseline min-max')
            plt.axvline(x=0, linestyle='--', color='k')
            plt.xlabel('Time(s)', fontsize=18)
            xticklabels = np.arange(-3, 6).tolist()
            xticklabels[3] = 'SZ'
            axes.set_xticklabels(xticklabels)
            plt.ylabel('max(Eigencentrality)', fontsize=16)
            # plt.ylim([0.3, 0.6])
            plt.title('Eigencentrality ({})'.format(band_name), fontsize=16)
            plt.legend(loc='upper right', fontsize=16)

            plt.scatter(t_axis[cross_indices], max_event_vals[cross_indices], marker='x', c='r')
            print('Saving figure to {}'.format(utils.namebase(output_fname)))
            plt.savefig(output_fname)
            utils.copy_file(output_fname, op.join(band_fol, figure_name))
            plt.close()

    return scores


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



def calc_scores(subject, files_dict, graph_func, bands, do_plot=False):
    scores = {}
    for event_name in ['ictal', 'IED', 'baseline']:
        scores[event_name] = {}
        for band_name in bands.keys():
            scores[event_name][band_name] = []

    for event_name in ['ictal', 'IED']:
        for sz_fname in files_dict[event_name]:
            score = plot_all_files_graph_max(
                subject, files_dict['baseline'], sz_fname, graph_func, bands, '', do_plot, overwrite=True)
            for band_name in bands.keys():
                if band_name in score:
                    scores[event_name][band_name].append(score[band_name])

    for sz_fname in files_dict['baseline']:
        new_baseline = utils.remote_items_from_list(files_dict['baseline'], [sz_fname])
        score = plot_all_files_graph_max(
            subject, new_baseline, sz_fname, graph_func, bands, 'baseline_{}'.format(utils.namebase(sz_fname)),
            do_plot, overwrite=True)
        if score is None:
            continue
        for band_name in bands.keys():
            if band_name in score:
                scores['baseline'][band_name].append(score[band_name])

    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity', graph_func, 'scores'))
    for band_name in bands.keys():
        all_scores = scores['ictal'][band_name] + scores['IED'][band_name] + scores['baseline'][band_name]
        if len(band_name) == 0:
            print('No results for {}!'.format(band_name))
            continue
        x = range(len(all_scores))
        names = ['ictal'] * len(files_dict['ictal']) + ['IED'] * len(files_dict['IED']) + \
                ['baseline'] * len(files_dict['baseline'])
        plt.bar(x, all_scores)
        plt.xticks(x, names, rotation='vertical')
        plt.savefig(op.join(output_fol, '{}_scores.jpg'.format(band_name)))
        plt.close()


def init_nmr01391():
    subject = 'nmr01391'
    remote_subject_dir = find_remote_subject_dir(subject)
    meg_fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/1391' # '/homes/5/npeled/space1/MEG/nmr00857/clips'
    bad_channels = ['MEG{}'.format(c) for c in [
        '0113', '1532', '1623', '2042', '1912', '2032', '2522', '0642', '0121', '1421', '1221', '1023',
        '0741', '1022', '1242']]
    fwd_usingMEG, fwd_usingEEG = True, False
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
    fwd_usingMEG, fwd_usingEEG = True, True
    raw_fname = op.join(meg_fol, '5241495_01_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, '5241495_roomnoise_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def find_remote_subject_dir(subject):
    remote_subjects_dir = '/space/megraid/clinical/MEG-MRI/seder/freesurfer'
    if not op.isdir(op.join(remote_subjects_dir, subject)):
        remote_subjects_dir = SUBJECTS_DIR
    remote_subject_dir = op.join(remote_subjects_dir, subject)
    if not op.isdir(remote_subject_dir):
        raise Exception('No reocon-all files!')
    return remote_subject_dir


if __name__ == '__main__':
    atlas = 'laus125'
    graph_func = 'eigenvector_centrality'
    connectivity_method = 'mi' # Mutual information
    bands = dict(all=[1, 120], delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    overwrite = True
    n_jobs = utils.get_n_jobs(-5)
    n_jobs = n_jobs if n_jobs > 1 else 1
    print('n_jobs = {}'.format(n_jobs))

    subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr00857()
    # subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01391()
    fwd_usingMEG, fwd_usingEEG = False, True
    modality = meg.get_modality(fwd_usingMEG, fwd_usingEEG)

    fif_files, files_dict = [], {}
    for subfol in ['ictal', 'baseline', 'IED']:
        files = glob.glob(op.join(meg_fol, subfol, '*.fif'))
        fif_files += files
        files_dict[subfol] = files

    # calc_anatomy(subject, atlas, remote_subject_dir, n_jobs)
    # calc_fwd_inv(
    #     subject, raw_fname, bad_channels, empty_room_fname, remote_subject_dir, fwd_usingMEG, fwd_usingEEG,
    #     overwrite, n_jobs)
    # calc_labels_data(
    #     subject, fif_files, atlas, remote_subject_dir, bad_channels, fwd_usingMEG, fwd_usingEEG, overwrite, n_jobs)
    # calc_connectivity(subject, atlas, connectivity_method, fif_files, bands, modality, overwrite, n_jobs)
    analyze_graphs(subject, fif_files, graph_func, bands, modality, overwrite, n_jobs)
    # calc_scores(subject, files_dict, graph_func, bands, do_plot=True)


