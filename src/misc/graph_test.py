import numpy as np
import os.path as op
import networkx as nx
from src.utils import utils
import time
import matplotlib.pyplot as plt

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def plot_con(subject, con_name):
    fol = op.join(MMVT_DIR, subject, 'connectivity')
    con = np.load(op.join(fol, '{}.npy'.format(con_name))).squeeze()
    t_axis = np.linspace(-2, 5, con.shape[2])
    plt.plot(t_axis, con[0].T)
    plt.title(con_name)
    plt.show()


def calc_measures(subject, con_name, func_name, n_jobs=4):
    fol = op.join(MMVT_DIR, subject, 'connectivity')
    print('Loading {}'.format(op.join(fol, '{}.npy'.format(con_name))))
    con = np.load(op.join(fol, '{}.npy'.format(con_name))).squeeze()
    # names = np.load(op.join(fol, 'labels_names.npy'))
    T = con.shape[2]
    con[con < np.percentile(con, 99)] = 0
    indices = np.array_split(np.arange(T), n_jobs)
    chunks = [(con, indices_chunk) for indices_chunk in indices]
    results = utils.run_parallel(calc_closeness_centrality, chunks, n_jobs)
    first = True
    for vals_chunk, times_chunk in results:
        if first:
            values = np.zeros((len(vals_chunk[0]), T))
            first = False
        values[:, times_chunk] = vals_chunk.T
    print('{}: min={}, max={}, mean={}'.format(con_name, np.min(values), np.max(values), np.mean(values)))
    np.save(op.join(fol, '{}_{}.npy'.format(con_name, func_name)), values)


def calc_closeness_centrality(p):
    con, times_chunk = p
    vals = []
    now = time.time()
    for run, t in enumerate(times_chunk):
        utils.time_to_go(now, run, len(times_chunk), 10)
        con_t = con[:, :, t]
        g = nx.from_numpy_matrix(con_t)
        # x = nx.closeness_centrality(g)
        x = np.degree_centrality(g)
        vals.append([x[k] for k in range(len(x))])
    vals = np.array(vals)
    return vals, times_chunk


def plot_values(subject, con_name, func_name, ma_win_size=10):
    vals = np.load(op.join(MMVT_DIR, subject, 'connectivity', '{}_{}.npy'.format(con_name, func_name)))
    # inds = np.argsort(np.max(vals, axis=1) - np.min(vals, axis=1))[::-1]
    # vals = vals[inds[:10]]
    # vals = utils.moving_avg(vals, ma_win_size)
    t_axis = np.linspace(-2, 5, vals.shape[1])
    # plt.plot(t_axis, np.diff(vals).T)
    plt.plot(t_axis, vals.T)
    plt.title(con_name)
    plt.show()


if __name__ == '__main__':
    n_jobs = utils.get_n_jobs(-5)
    print('n_jobs: {}'.format(n_jobs))
    subject = 'nmr00857'
    func_name = 'degree_centrality', #'closeness_centrality' # 'clustering'
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    for band_name in bands.keys():
        con_name = 'meg_{}_mi'.format(band_name)
        # plot_con(subject, con_name)
        calc_measures(subject, con_name, func_name, n_jobs)
        # plot_values(subject, con_name, func_name)