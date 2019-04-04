import os.path as op
import glob
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import math

from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import meg
from src.preproc import connectivity

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def main(subject, atlas, connectivity_method, graph_func, remote_subject_dir, files, n_jobs=4):
    meg_fol = op.join(MMVT_DIR, subject, 'meg')
    for fname in files:
        file_name = utils.namebase(fname)
        files_fol = utils.make_dir(op.join(meg_fol, file_name))
        files_exist = \
            utils.both_hemi_files_exist(op.join(files_fol, '{}_all-dSPM-{}.stc'.format(subject, '{hemi}'))) and \
            utils.both_hemi_files_exist(op.join(files_fol, 'labels_data_epilepsy_laus125_dSPM_mean_flip_{hemi}.npz'))
        if not files_exist:
            args = meg.read_cmd_args(dict(
                subject=subject,
                task='epilepsy',
                function='calc_stc,calc_labels_avg_per_condition',
                atlas=atlas,
                evo_fname = fname,
                remote_subject_dir=remote_subject_dir,
                overwrite_labels_data=True,
                n_jobs=n_jobs
            ))
            meg.call_main(args)
        analyse_connectivity(subject, connectivity_method, graph_func, file_name, atlas, n_jobs)
        for hemi in utils.HEMIS:
            utils.move_file(op.join(meg_fol, '{}_all-dSPM-{}.stc'.format(subject, hemi)), files_fol)
            utils.move_file(
                op.join(meg_fol, 'labels_data_epilepsy_laus125_dSPM_mean_flip_{}.npz'.format(hemi)), files_fol)

    plot_all_files_graph_max(subject, files, graph_func)
    # save_sz_pick_values(subject, files, graph_func, atlas)


def analyse_connectivity(subject, connectivity_method, graph_func, file_name, atlas, n_jobs=4):
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    conn_fol = op.join(MMVT_DIR, subject, 'connectivity')
    output_fol = utils.make_dir(op.join(conn_fol, file_name))
    for band_name, band_freqs in bands.items():
        output_files = [f.format(band_name) for f in
                        ['meg_{}_laus125_mi_mean_flip_mean.npy', 'meg_{}_mi.npy', 'meg_{}_mi.npz']]
        files_exist = all([op.isfile(op.join(output_fol, f)) for f in output_files])
        if not files_exist:
            print('calc_meg_connectivity: {}'.format(band_name))
            con_args = connectivity.read_cmd_args(utils.Bag(
                subject=subject,
                atlas=atlas,
                function='calc_lables_connectivity',
                connectivity_modality='meg',
                connectivity_method=connectivity_method,
                windows_length=500,
                windows_shift=100,
                identifier=band_name,
                fmin=band_freqs[0],
                fmax=band_freqs[1],
                sfreq=2035, # clin MEG
                recalc_connectivity=True,
                n_jobs=n_jobs
            ))
            connectivity.call_main(con_args)
            for output_file_name in output_files:
                utils.move_file(op.join(conn_fol, output_file_name), output_fol)

        con_name = 'meg_{}_mi'.format(band_name)
        analyze_graph(subject, file_name, con_name, graph_func, n_jobs)
        plot_graph_values(subject, file_name, con_name, graph_func)


def analyze_graph(subject, file_name, con_name, graph_func, n_jobs=4):
    fol = op.join(MMVT_DIR, subject, 'connectivity', file_name)
    output_fname = op.join(fol, '{}_{}.npy'.format(con_name, graph_func))
    if op.isfile(output_fname):
        return
    print('Loading {}'.format(op.join(fol, '{}.npy'.format(con_name))))
    con = np.load(op.join(fol, '{}.npy'.format(con_name))).squeeze()
    T = con.shape[2]
    con[con < np.percentile(con, 99)] = 0
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


def plot_all_files_graph_max(subject, files_names, func_name):
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity', func_name))
    sfreq = 2035
    windows_length = 500
    half_window = (1 /sfreq) * (windows_length / 2) # In seconds
    for band_name in bands.keys():
        con_name = 'meg_{}_mi'.format(band_name)
        output_fname =  op.join(output_fol, '{}_{}.jpg'.format(con_name, func_name))
        # if op.isfile(output_fname):
        #     continue
        baseline_values = []
        all_values = []
        # sz_ind = 0
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111)
        for fname in files_names:
            file_name = utils.namebase(fname)
            is_sz = 'SZ' in file_name
            input_fname = op.join(MMVT_DIR, subject, 'connectivity', file_name, '{}_{}.npy'.format(con_name, func_name))
            vals = np.load(input_fname)
            # t_axis = np.linspace(-2, 5, vals.shape[1])
            t_axis = np.linspace(-2 + half_window, 5 - half_window, vals.shape[1])
            max_vals = np.max(vals, axis=0)
            all_values.append(max_vals)
            if is_sz:
                # sz_ind += 1
                # if 'start' in file_name:
                #     max_vals[np.where(t_axis < 0)[0]] = np.nan
                # else:
                plt.plot(t_axis, max_vals, label='SZ')# {}'.format(sz_ind))
            else:
                baseline_values.append(max_vals)
        baseline_values_max = np.array(baseline_values).max(axis=0)
        baseline_values_mean = np.array(baseline_values).mean(axis=0)
        baseline_values_std = np.array(baseline_values).std(axis=0)
        plt.plot(t_axis, baseline_values_max, '--', label='No SZ max')
        plt.plot(t_axis, baseline_values_mean, '--', label='No SZ mean')
        plt.fill_between(t_axis, baseline_values_mean - 2 * baseline_values_std, baseline_values_mean + 2 * baseline_values_std,
                         color='#539caf', alpha=0.4, label='No SZ' + r'$ \pm 2\sigma$')
        plt.axvline(x=0, linestyle='--', color='k')
        plt.xlabel('Time(s)', fontsize=18)
        xticklabels = np.arange(-3, 6).tolist()
        xticklabels[3] = 'SZ'
        axes.set_xticklabels(xticklabels)
        plt.ylabel('max(Eigencentrality)', fontsize=16)
        # plt.ylim([0.3, 0.6])
        plt.title('Eigencentrality ({})'.format(band_name), fontsize=16)
        plt.legend(loc='upper right', fontsize=16)
        print('Saving figure to {}'.format(output_fname))
        plt.savefig(output_fname)
        plt.close()

        # all_values = np.array(all_values)
        # plt.figure()



def save_sz_pick_values(subject, files_names, func_name, atlas):
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
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


if __name__ == '__main__':
    subject = 'nmr00857'
    atlas = 'laus125'
    graph_func = 'eigenvector_centrality'
    connectivity_method = 'mi' # Mutual information
    n_jobs = utils.get_n_jobs(-5)
    print('n_jobs = {}'.format(n_jobs))
    remote_subject_dir = '/space/megraid/clinical/MEG-MRI/seder/freesurfer/nmr00857'
    # fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/run1_NoFilter'
    fol = '/homes/5/npeled/space1/MEG/nmr00857/clips'
    files = glob.glob(op.join(fol, '*.fif'))
    main(subject, atlas, connectivity_method, graph_func, remote_subject_dir, files, n_jobs)
    # plot_con(subject, files, 'beta')