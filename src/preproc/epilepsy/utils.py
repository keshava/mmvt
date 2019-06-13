import numpy as np
from numpy.ma import masked_array
import glob
import os.path as op
import mne

from src.utils import utils


def calc_powers_abs_minmax(powers, label_norm_powers_files=None, both_min_and_max=True):

    # -- Using mean of lablels
    # labels_norm_powers = np.concatenate(
    #     [np.mean(np.load(f), axis=0, keepdims=True) for f in label_norm_powers_files])
    # powers_min = np.min(labels_norm_powers, axis=0)
    # powers_max = np.max(labels_norm_powers, axis=0)

    # -- Using the same vertice
    # min_vertice = np.argmin(np.min(powers, axis=(1, 2)))
    # max_vertice = np.argmax(np.max(powers, axis=(1, 2)))
    # powers_max = np.max(powers, axis=(1, 2))[max_vertice]
    # powers_min = np.min(powers, axis=(1, 2))[min_vertice]
    # minmax_vertice = max_vertice # if powers_max > abs(powers_min) else min_vertice
    # print('max vertice {} -> {}'.format(minmax_vertice, powers_max))
    # print(np.unravel_index(powers[minmax_vertice].argmax(), powers[minmax_vertice].shape))
    # return powers[minmax_vertice]

    # -- Can be different vertice for each time and freq
    max_vertices = powers.reshape(powers.shape[0], -1).argmax(0).reshape((powers.shape[1], -1))
    min_vertices = powers.reshape(powers.shape[0], -1).argmin(0).reshape((powers.shape[1], -1))
    powers_min = np.min(powers, axis=0)  # over vertices
    powers_max = np.max(powers, axis=0)  # over vertices
    print('minmin: {}, maxmax: {}'.format(np.min(powers_min), np.max(powers_max)))
    if both_min_and_max:
        return powers_min, powers_max, min_vertices, max_vertices
    else:
        min_indices = np.where(np.abs(powers_min) > powers_max)
        powers_abs_minmax = powers_max
        powers_abs_minmax[min_indices] = powers_min[min_indices]
        return powers_abs_minmax


def concatenate_powers(fol, return_file_names=False):
    print('Concatenate powers in {}'.format(fol))
    powers_files = glob.glob(op.join(fol, 'epilepsy_*_induced_power.npy'))
    if len(powers_files) == 0:
        print('No files in {}!'.format(fol))
        return None
    # if len(powers_files) != 62: # Should calc number of lables
    #     print('{}: Not all the files were created!'.format(fol))
    #     return None
    powers = np.concatenate([np.load(powers_fname).astype(np.float32) for powers_fname in powers_files])
    if return_file_names:
        return powers, powers_files
    else:
        return powers


def calc_masked_negative_and_positive_powers(norm_powers_min, norm_powers_max, percentiles=[5, 95]):
    minmax_powers = np.zeros(norm_powers_max.shape)
    min_inds = np.where(norm_powers_min < np.percentile(norm_powers_min, percentiles[0]))
    max_inds = np.where(norm_powers_max > np.percentile(norm_powers_max, percentiles[1]))
    minmax_powers[max_inds] = norm_powers_max[max_inds]
    minmax_powers[min_inds] = norm_powers_min[min_inds]
    negative_powers = masked_array(minmax_powers, minmax_powers < np.percentile(norm_powers_min, percentiles[0]))
    positive_powers = masked_array(minmax_powers, minmax_powers > np.percentile(norm_powers_max, percentiles[1]))
    return negative_powers, positive_powers


def get_window_times(window_fname, downsample=2):
    evoked = mne.read_evokeds(window_fname)[0]
    times = evoked.times if len(evoked.times) % downsample == 0 else \
        evoked.times[:-(downsample - 1)]
    return utils.downsample(times, downsample)


def nans(shape, dtype=np.float32):
    x = np.empty(shape, dtype)
    x.fill(np.nan)
    return x
