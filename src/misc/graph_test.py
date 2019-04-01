import numpy as np
import os.path as op
import networkx as nx
from src.utils import utils
import time
import matplotlib.pyplot as plt


def calc_measures(fol, graph_func, n_jobs=4):
    con = np.load(op.join(fol, 'meg_pli.npy')).squeeze()
    names = np.load(op.join(fol, 'labels_names.npy'))
    T = con.shape[2]

    indices = np.array_split(np.arange(T), n_jobs)
    chunks = [(con, indices_chunk) for indices_chunk in indices]
    results = utils.run_parallel(calc_closeness_centrality, chunks, n_jobs)
    first = True
    for vals_chunk, times_chunk in results:
        if first:
            values = np.zeros((len(vals_chunk[0]), T))
            first = False
        values[:, times_chunk] = vals_chunk.T
    np.save(op.join(fol, 'clustering.npy'), values)


def calc_closeness_centrality(p):
    con, times_chunk = p
    vals = []
    now = time.time()
    for run, t in enumerate(times_chunk):
        utils.time_to_go(now, run, len(times_chunk), 10)
        g = nx.from_numpy_matrix(con[:, :, t])
        # clos = nx.closeness_centrality(g)
        x = np.clustering(g)
        vals.append([x[k] for k in range(len(x))])
        # vals.append([k for k in range(219)])
    vals = np.array(vals)
    return vals, times_chunk


def plot_values(fol):
    vals = np.load(op.join(fol, 'closeness_centrality.npy'))
    t_axis = np.linspace(-2, 5, vals.shape[1] - 1)
    plt.plot(t_axis, np.diff(vals).T)
    plt.show()


if __name__ == '__main__':
    n_jobs = utils.get_n_jobs(-5)
    print('n_jobs: {}'.format(n_jobs))
    fol = '/homes/5/npeled/space1/mmvt/nmr00857/connectivity/'
    calc_measures(fol, n_jobs)
    # plot_values(fol)