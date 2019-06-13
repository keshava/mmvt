import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
import glob
import os.path as op
import mne
import numpy as np
from numpy.ma import masked_array
import re

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels, overwrite_inv=False,
                 overwrite_fwd=False):
    # python -m src.preproc.eeg -s nmr00857 -f calc_inverse_operator,make_forward_solution
    #     --overwrite_inv 0 --overwrite_fwd 0 -t epilepsy
    #     --raw_fname  /autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123/5241495_01_raw.fif
    #     --empty_fname /autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123/5241495_roomnoise_raw.fif
    #     --use_empty_room_for_noise_cov 1
    #     --bad_channels EEG061,EEG02,EEG042,MEG0112,MEG0113
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_inverse_operator,make_forward_solution',
        task='epilepsy',
        inv_fname=op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality)),
        fwd_fname=op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality)),
        fwd_usingEEG=modality in ['eeg', 'meeg'],
        overwrite_inv=overwrite_inv,
        overwrite_fwd=overwrite_fwd,
        use_empty_room_for_noise_cov=True,
        bad_channels=bad_channels,
        raw_fname=raw_fname,
        empty_fname=empty_fname,
    ))
    module.call_main(args)


def check_inv_fwd(subject, modality, run_num):
    import mne.minimum_norm
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    fwd_fname = op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality))
    fwd = mne.read_forward_solution(fwd_fname)
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    fwd_meg_channels = [c for c in fwd['sol']['row_names'] if c.startswith('MEG')]
    fwd_eeg_channels = [c for c in fwd['sol']['row_names'] if c.startswith('EEG')]
    inv_meg_channels = [c for c in inv['info']['ch_names'] if c.startswith('MEG')]
    inv_eeg_channels = [c for c in inv['info']['ch_names'] if c.startswith('EEG')]
    print('{}: using {}/{} EEG sensors and {}/{} MEG sensors'.format(
        modality, len(inv_eeg_channels), len(fwd_eeg_channels), len(inv_meg_channels), len(fwd_meg_channels)))


def calc_amplitude(subject, modality, run_num, windows_fnames, inverse_method='dSPM', overwrite=False, n_jobs=4):
    params = [(subject, window_fname, modality, run_num, windows_fnames, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_amplitude_parallel, params, n_jobs)


def _calc_amplitude_parallel(p):
    # python3 -m src.preproc.meg -s nmr00857 -f calc_stc -i dSPM -t epilepsy
    #   --evo_fname /autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/43.9s.fif
    #   --overwrite_stc 1
    subject, window_fname, modality, run_num, windows_fnames, inverse_method, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_stc',
        task='epilepsy',
        inverse_method=inverse_method,
        inv_fname=op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality)),
        fwd_fname=op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality)),
        fwd_usingEEG=modality in ['eeg', 'meeg'],
        evo_fname=window_fname,
        overwrite_stc=overwrite,
        n_jobs=1,
    ))
    module.call_main(args)


def calc_amplitude_zvals(subject, windows_fnames, baseline_name, modality, from_index=None, to_index=None,
                         inverse_method='dSPM', parallel=True, overwrite=False):
    params = [(subject, modality, window_fname, baseline_name, from_index, to_index, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_amplitude_zvals_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_amplitude_zvals_parallel(p):
    # python3 -m src.preproc.meg -s nmr00857 -f calc_stc_zvals --stc_name nmr00857-epilepsy-dSPM-meeg-43.9s
    #   --baseline_stc_name nmr00857-epilepsy-dSPM-meeg-37.3_BGprSzs --use_abs 1 --overwrite_stc 1
    subject, modality, window_fname, baseline_name, from_index, to_index, inverse_method, overwrite = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    stc_template = '{}-epilepsy-{}-{}-{}{}'.format(subject, inverse_method, modality, '{window}', '{suffix}')
    window_stc_name = stc_template.format(window=window, suffix='')
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_stc_zvals',
        task='epilepsy',
        stc_name=window_stc_name,
        baseline_stc_name=stc_template.format(window=baseline_name, suffix=''),
        stc_zvals_name=stc_template.format(window=window, suffix='_amplitude-zvals'),
        from_index=from_index,
        to_index=to_index,
        use_abs=1,
        overwrite_stc=overwrite
    ))
    module.call_main(args)


def calc_sensors_power(subject, windows_fnames, modality, inverse_method='dSPM', bad_channels=[],
                       high_gamma_max=120, downsample=2, parallel=False, overwrite=False):
    params = [(subject, window_fname, modality, inverse_method, bad_channels, downsample, high_gamma_max, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_sensors_power_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_sensors_power_parallel(p):
    from mne.time_frequency import tfr_array_morlet

    subject, window_fname, modality, inverse_method, bad_channels, downsample, high_gamma_max, overwrite = p

    root_dir = utils.make_dir(op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject))
    output_fname_template = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_power.npy'.format(
        subject, inverse_method, modality, '{window}'))
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    bad_channels = bad_channels.split(',')
    n_cycles = freqs / 2.

    window = utils.namebase(window_fname)
    output_fname = output_fname_template.format(window=window)
    if op.isfile(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
    evoked = mne.read_evokeds(window_fname)[0]
    if modality == 'eeg':
        picks = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=bad_channels)
    elif modality == 'meg':
        picks = mne.pick_types(evoked.info, meg=True, eeg=False, exclude=bad_channels)
    elif modality == 'meeg':
        picks = mne.pick_types(evoked.info, meg=True, eeg=True, exclude=bad_channels)
    else:
        raise Exception('Wrong modality!')

    evoked_data = evoked.data[np.newaxis, picks, :]
    powers = tfr_array_morlet(
        evoked_data, sfreq=evoked.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output='power')
    powers = np.squeeze(powers)
    if powers.shape[2] % 2 == 1:
        powers = powers[:, :, :-1]
    if downsample > 1:
        powers = utils.downsample_3d(powers, downsample)
    powers_db = 10 * np.log10(powers)  # dB/Hz should be baseline corrected!!!
    print('Saving {}'.format(output_fname))
    np.save(output_fname, powers_db.astype(np.float16))


def plot_sensors_powers(subject, windows_fnames, baseline_window_fname, modality, inverse_method='dSPM',
                        high_gamma_max=120, percentiles=[5, 95], sig_threshold= 2, overwrite=False, parallel=True):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    input_fname_template = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_power.npy'.format(
        subject, inverse_method, modality, '{window}'))
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'sensors-power-spectrum'))
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-sensors-power.jpg'.format(
            subject, inverse_method, modality, '{window}'))

    baseline_window = utils.namebase(baseline_window_fname)
    if not op.isfile(input_fname_template.format(window=baseline_window)):
        print('No baseline powers! {}'.format(input_fname_template.format(window=baseline_window)))
        return
    baseline = np.load(input_fname_template.format(window=baseline_window)).astype(np.float32)
    baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
    baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
    baseline_mean_over_sensors = np.mean(baseline, axis=0)
    plot_power_spectrum(baseline_mean_over_sensors, figures_template.format(window=baseline_window),
                        remove_non_sig=False, high_gamma_max=high_gamma_max)
    for window_fname in windows_fnames:
        window = utils.namebase(window_fname)
        figure_fname = figures_template.format(window=window)
        # dividing by the mean of baseline values, taking the log, and
        #           dividing by the standard deviation of log baseline values
        #           ('zlogratio')
        if not op.isfile(input_fname_template.format(window=window)):
            print('No window powers! {}'.format(input_fname_template.format(window=window)))
            continue
        powers = np.load(input_fname_template.format(window=window)).astype(np.float32)
        norm_powers = (powers - baseline_mean) / baseline_std
        # powers_min = np.min(powers, axis=0) # over vertices
        # powers_max = np.max(powers, axis=0) # over vertices
        # min_indices = np.where(np.abs(powers_min) > powers_max)
        # powers_abs_minmax = powers_max
        # powers_abs_minmax[min_indices] = powers_min[min_indices]
        # plot_power_spectrum(powers_abs_minmax, figure_fname, high_gamma_max=high_gamma_max)

        norm_powers[np.where(np.abs(norm_powers) < sig_threshold)] = 0
        norm_powers_min, norm_powers_max = calc_powers_abs_minmax(norm_powers)
        # negative_powers, positive_powers = calc_masked_negative_and_positive_powers(
        #     norm_powers_min, norm_powers_max, percentiles)
        times = get_window_times(window_fname, downsample=2)
        plot_positive_and_negative_power_spectrum(
            norm_powers_min, norm_powers_max, times,  '{} {}'.format(modality, window),
            figures_template.format(window=window, method='minmax_two_layers'), high_gamma_max=high_gamma_max)
        # plot_power_spectrum_two_layers(
        #     positive_powers, negative_powers, times, '{} {}'.format(modality, window),
        #     figures_template.format(window=window, method='minmax_two_layers'), high_gamma_max=high_gamma_max)


def calc_induced_power(subject, run_num, windows_fnames, modality, inverse_method='dSPM', check_for_labels_files=True,
                       overwrite=False):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    output_fname = op.join(MMVT_DIR, 'eeg' if modality == 'eeg' else 'meg', '{}-epilepsy-{}-{}-{}_{}'.format(
        subject, inverse_method, modality, '{window}', '{band}'))
    for window_fname in windows_fnames:
        if all([utils.stc_exist(output_fname.format(window=utils.namebase(window_fname), band=band))
                for band in bands]) and not overwrite:
            continue
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, utils.namebase(window_fname)))
        if op.isdir(fol) and not check_for_labels_files:
            print('{} already exist'.format(fol))
            continue
        args = module.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            task='epilepsy',
            inverse_method=inverse_method,
            inv_fname=op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality)),
            fwd_fname=op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality)),
            calc_source_band_induced_power=True,
            fwd_usingEEG=modality in ['eeg', 'meeg'],
            evo_fname=window_fname,
            n_jobs=1,
            overwrite_stc=overwrite
        ))
        module.call_main(args)


def plot_norm_powers(subject, windows_fnames, baseline_window, modality, inverse_method='dSPM', figures_type='jpg',
                     calc_also_non_norm_powers=False, use_norm_labels_powers=False, overwrite=False):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    if use_norm_labels_powers:
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power_labels.npy'.format(
            subject, inverse_method, modality, '{window}'))
    else:
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
            subject, inverse_method, modality, '{window}'))
    not_norm_output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_minmax_power.npy'.format(
        subject, inverse_method, modality, '{window}'))
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    figs_fol_not_norm = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum-not-norm'))
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-induced_{}_norm_power.{}'.format(
            subject, inverse_method, modality, '{window}', '{method}', figures_type))
    figures_template_not_norm = op.join(figs_fol_not_norm, '{}-epilepsy-{}-{}-{}-induced_mean_norm_power.jpg'.format(
            subject, inverse_method, modality, '{window}'))
    if not all([op.isfile(output_fname.format(window=utils.namebase(window_fname)))
                for window_fname in windows_fnames]) or overwrite:
            #  or not op.isfile(figures_template.format(window=utils.namebase(baseline_window)))
        baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
                subject, inverse_method, modality, utils.namebase(baseline_window)))
        baseline = concatenate_powers(baseline_fol) # (vertices x freqs x time)
        if baseline is None:
            return
        baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the mean (over time) of log baseline values
        # baseline_mean_over_vertices = np.mean(baseline, axis=0)
        # plot_power_spectrum(baseline_mean_over_vertices, figures_template.format(window=utils.namebase(baseline_window)),
        #                     remove_non_sig=False)
    for window_fname in windows_fnames:
        window = utils.namebase(window_fname)
        window_output_fname = output_fname.format(window=window)
        window_not_norm_fname = not_norm_output_fname.format(window=window)
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(subject, inverse_method, modality, window))
        if not op.isfile(window_output_fname) or overwrite or \
                (not op.isfile(window_not_norm_fname) and calc_also_non_norm_powers):
            powers = concatenate_powers(fol)
            if calc_also_non_norm_powers:
                freqs_norm = np.max(powers, axis=(0, 2), keepdims=True)
                freqs_norm_powers = powers / freqs_norm
                powers_abs_minmax = calc_powers_abs_minmax(freqs_norm_powers)
                np.save(window_not_norm_fname, powers_abs_minmax)
            # dividing by the mean of baseline values, taking the log, and  dividing by the standard deviation of
            # log baseline values ('zlogratio')
            norm_powers = (powers - baseline_mean) / baseline_std
            # if use_norm_labels_powers:
            #     fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            #         subject, inverse_method, modality, window))
            #     label_norm_powers = glob.glob(op.join(fol, 'epilepsy_*_induced_norm_power.npy'))
            # else:
            #     label_norm_powers = None

            norm_powers_abs_minmax = calc_powers_abs_minmax(norm_powers, both_min_and_max=False)#label_norm_powers)
            np.save(window_output_fname, norm_powers_abs_minmax)
            norm_powers_min, norm_powers_max, min_vertices, max_vertices = calc_powers_abs_minmax(norm_powers, both_min_and_max=True)
            np.savez(window_output_fname.replace('npy', 'npz'), min=norm_powers_min, max=norm_powers_max,
                     min_vertices=min_vertices, max_vertices=max_vertices)
        else:
            norm_powers_abs_minmax = np.load(window_output_fname)
            d = np.load(window_output_fname.replace('npy', 'npz'))
            norm_powers_min, norm_powers_max = d['min'], d['max']
            if calc_also_non_norm_powers:
                powers_abs_minmax = np.load(window_not_norm_fname)

        percentiles = [5, 95]
        negative_powers, positive_powers = calc_masked_negative_and_positive_powers(norm_powers_min, norm_powers_max, percentiles)

        # x1, x2 = nans(norm_powers_max.shape), nans(norm_powers_max.shape)
        # x1[max_inds] = norm_powers_max[max_inds]
        # x2[min_inds] = norm_powers_min[min_inds]
        times = get_window_times(window_fname, downsample=2)

        plot_power_spectrum_two_layers(positive_powers, negative_powers, times, '{} {}'.format(modality, window),
                                       figures_template.format(window=window, method='minmax_two_layers'))

        # plot_power_spectrum(norm_powers_abs_minmax, figures_template.format(window=window, method='minmax'), baseline_correction=False)
        # plot_power_spectrum(norm_powers_min, figures_template.format(window=window, method='min'), baseline_correction=False)
        # plot_power_spectrum(norm_powers_max, figures_template.format(window=window, method='max'), baseline_correction=False)
        if calc_also_non_norm_powers:
            plot_power_spectrum(
                powers_abs_minmax, figures_template_not_norm.format(window=window), vmax=1,
                baseline_correction=False, remove_non_sig=False)


def get_window_times(window_fname, downsample=2):
    evoked = mne.read_evokeds(window_fname)[0]
    times = evoked.times if len(evoked.times) % downsample == 0 else \
        evoked.times[:-(downsample - 1)]
    return utils.downsample(times, downsample)


def nans(shape, dtype=np.float32):
    x = np.empty(shape, dtype)
    x.fill(np.nan)
    return x

def plot_norm_powers_per_label(subject, windows_fnames, baseline_window, modality, inverse_method='dSPM',
                               calc_also_non_norm_powers=False, overwrite=False, n_jobs=4):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(baseline_window)))

    baseline_labels = glob.glob(op.join(baseline_fol, 'epilepsy_*_induced_power.npy'))
    indices = np.array_split(np.arange(len(baseline_labels)), n_jobs)
    chunks = [([baseline_labels[ind] for ind in chunk_indices], subject, windows_fnames, inverse_method, modality,
               baseline_window, calc_also_non_norm_powers, overwrite) for chunk_indices in indices]
    utils.run_parallel(_plot_norm_powers_per_label_parallel, chunks, n_jobs)


def _plot_norm_powers_per_label_parallel(p):
    (baseline_labels_chunk, subject, windows_fnames, inverse_method, modality, baseline_window,
     calc_also_non_norm_powers, overwrite) = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'labels-power-spectrum'))
    figs_no_norm_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'labels-power-spectrum-no-norm'))

    for label_baseline_fname in baseline_labels_chunk:
        label_name = utils.namebase(label_baseline_fname).split('_')[1]
        label_figs_fol = utils.make_dir(op.join(figs_fol, label_name))
        label_figs_no_norm_fol = utils.make_dir(op.join(figs_no_norm_fol, label_name))
        figures_template = op.join(label_figs_fol, '{}-epilepsy-{}-{}-{}-{}-induced_power.jpg'.format(
            subject, inverse_method, modality, '{window}', label_name))
        figures_template_no_norm = op.join(label_figs_no_norm_fol, '{}-epilepsy-{}-{}-{}-{}-induced_power.jpg'.format(
            subject, inverse_method, modality, '{window}', label_name))
        baseline = np.load(label_baseline_fname).astype(np.float32)
        baseline_std = np.std(baseline, axis=2, keepdims=True)  # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True)  # the standard deviation (over time) of log baseline values
        _calc_label_power_over_windows(
            label_name, modality, baseline_mean, baseline_std, windows_fnames, root_dir, figures_template,
            figures_template_no_norm, calc_also_non_norm_powers, overwrite)


def _calc_label_power_over_windows(label_name, modality, baseline_mean, baseline_std, windows_fnames, root_dir,
                                   figures_template, figures_template_no_norm, calc_also_non_norm_powers, overwrite):
    for window_fname in windows_fnames:
        window = utils.namebase(window_fname)
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, window))
        label_powers_fname = op.join(fol, 'epilepsy_{}_induced_power.npy'.format(label_name))
        label_norm_powers_fname = op.join(fol, 'epilepsy_{}_induced_norm_power.npy'.format(label_name))
        if op.isfile(figures_template.format(window=window)) and op.isfile(label_norm_powers_fname) and not overwrite:
            print('{} already exist'.format(figures_template.format(window=window)))
            return
        if op.isfile(label_norm_powers_fname) and not overwrite:
            norm_powers = np.load(label_norm_powers_fname)
        else:
            label_powers = np.load(label_powers_fname).astype(np.float32)
            # if calc_also_non_norm_powers and not op.isfile(figures_template_no_norm.format(window=window)) \
            #         and not overwrite:
            #     freqs_norm = np.max(label_powers, axis=(0, 2), keepdims=True)
            #     freqs_norm_powers = label_powers / freqs_norm
            #     powers_abs_minmax = calc_powers_abs_minmax(freqs_norm_powers)
            #     plot_power_spectrum(
            #         powers_abs_minmax, figures_template_no_norm.format(window=window), vmax=1,
            #         baseline_correction=False, remove_non_sig=False)
            norm_powers = (label_powers - baseline_mean) / baseline_std
            print('Saving {} norm powers to {}'.format(label_name, label_norm_powers_fname))
            np.save(label_norm_powers_fname, norm_powers.astype(np.float16))
        # norm_powers_abs_minmax = calc_powers_abs_minmax(norm_powers)
        # plot_power_spectrum(norm_powers_abs_minmax, figures_template.format(window=window))


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
    # max_vertices = powers.reshape(powers.shape[0], -1).argmax(0).reshape((powers.shape[1], -1))
    # min_vertices = powers.reshape(powers.shape[0], -1).argmin(0).reshape((powers.shape[1], -1))
    powers_min = np.min(powers, axis=0)  # over vertices
    powers_max = np.max(powers, axis=0)  # over vertices
    print('minmin: {}, maxmax: {}'.format(np.min(powers_min), np.max(powers_max)))
    if both_min_and_max:
        return powers_min, powers_max #, min_vertices, max_vertices
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


def calc_max_powers(subject, windows_fnames, modality, inverse_method='dSPM', overwrite=False, parallel=True):
    params = [(subject, window_fname, modality, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_max_powers_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_max_powers_parallel(p):
    subject, window_fname, modality, inverse_method, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_max_power.npy'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if op.isfile(output_fname) and not overwrite:
        return
    fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if not op.isdir(fol):
        print('{} does not exist!'.format(fol))
        return
    powers_files = glob.glob(op.join(fol, 'epilepsy_*_induced_power.npy'))
    if len(powers_files) != 62: # Should calc number of lables
        print('{}: Not all the files were created!'.format(fol))
        return
    print('Calculating max power for {}'.format(fol))
    max_powers = np.max(concatenate_powers(fol), axis=0)
    print('Saving to {}'.format(output_fname))
    np.save(output_fname, max_powers)


def plot_max_powers(subject, windows_fnames, modality, inverse_method='dSPM', overwrite=False, parallel=True):
    params = [(subject, window_fname, modality, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_plot_max_powers_parllel, params, len(windows_fnames) if parallel else 1)


def _plot_max_powers_parllel(p):
    subject, window_fname, modality, inverse_method, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_max_power.npy'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if not op.isfile(fname):
        print('Can\'t find {}!'.format(fname))
        return
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    figure_fname = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-induced_max_power.jpg'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    # if op.isfile(figure_fname) and not overwrite:
    #     return
    powers = np.load(fname)
    plot_power_spectrum(powers, figure_fname)


@utils.tryit()
def plot_power_spectrum(powers, figure_fname, remove_non_sig=True, vmax=None, vmin=None, baseline_correction=True,
                        calc_dt=False, high_gamma_max=120):
    # powers: (freqs x time)
    from src.utils import color_maps_utils as cmu
    BuPu_YlOrRd_cm = cmu.create_BuPu_YlOrRd_cm()
    F, T = powers.shape
    powers = powers.astype(np.float32)

    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    print('Plotting {}'.format(figure_fname))
    if baseline_correction:
        powers -= np.mean(powers[:, 0])
    if calc_dt:
        powers = np.diff(powers, axis=1)
        times = np.arange(powers.shape[1] - 1)
    else:
        times = np.arange(powers.shape[1])

    # high_gamma_top = 120 if powers.shape[0] == 51 else 125
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    if powers.shape[0] != len(freqs):
        print('powers.shape[0] != len(freqs)!!!')
        return
    _vmax, _vmin = np.max(powers), np.min(powers)
    print('vmin: {}, vmax: {}'.format(_vmin, _vmax))
    if _vmin > 0:
        cmap = 'YlOrRd'
        vmax = _vmax if vmax is None else vmax
        vmin = _vmin if vmin is None else vmin
    elif _vmax < 0:
        cmap = 'BuPu'
        vmax = _vmax if vmax is None else vmax
        vmin = _vmin if vmin is None else vmin
    else:
        cmap = BuPu_YlOrRd_cm
        if vmax is None and vmin is None:
            maxmin = max(map(abs, [_vmax, _vmin]))
        elif vmax is None:
            maxmin = abs(vmin)
        elif vmin is None:
            maxmin = vmax
        vmin, vmax = -maxmin, maxmin

    plt.subplot(211)
    clean_powers = powers.copy()
    if remove_non_sig:
        clean_powers[np.where(np.abs(powers) < 2)] = 0
    plt.imshow(np.flip(clean_powers, 0), vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest',
               extent=extents(times) + extents(freqs), cmap=cmap)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time points')
    plt.colorbar()
    # plt.tight_layout()
    plt.subplot(212)
    for band_name, band_freqs in bands.items():
        idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        band_power = np.mean(powers[idx, :], axis=0)
        plt.plot(band_power.T, label=band_name)
    plt.xlim([0, T])
    plt.legend()
    plt.savefig(figure_fname, dpi=300)
    plt.close()


def plot_positive_and_negative_power_spectrum(
        powers_negative, powers_positive, times, title='', figure_fname='',
        only_power_spectrum=True, high_gamma_max=120):

    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    bands = dict(delta=[1, 4], gamma=[30, 55], # theta=[4, 8], alpha=[8, 15], beta=[15, 30],
                 high_gamma=[55, 100], hfo=[100, high_gamma_max])
    min_t, max_t = round(times[0]), round(times[-1])

    fig, axs = plt.subplots(2, 2)
    pos_powers_ax, neg_powers_ax = axs[0, 0], axs[0, 1]
    pos_graph_ax, neg_graph_ax = axs[1, 0], axs[1, 1]
    im1 = _plot_powers(powers_negative, neg_powers_ax, times, high_gamma_max)
    # cba = plt.colorbar(im1, ax=neg_powers_ax,  shrink=0.25)
    im2 = _plot_powers(powers_positive, pos_powers_ax, times, high_gamma_max)
    # cbb = plt.colorbar(im2, ax=pos_graph_ax, shrink=0.25)

    for band_name, band_freqs in bands.items():
        idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        band_power = np.mean(powers_negative[idx, :], axis=0)
        neg_graph_ax.plot(times, band_power.T, label=band_name)
    neg_graph_ax.set_xlim([min_t, max_t])
    # neg_graph_ax.legend()

    for band_name, band_freqs in bands.items():
        idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        band_power = np.mean(powers_positive[idx, :], axis=0)
        pos_graph_ax.plot(times, band_power.T, label=band_name)
    pos_graph_ax.set_xlim([min_t, max_t])
    pos_graph_ax.legend()
    plt.suptitle(title)

    if figure_fname != '':
        print('Saving figure to {}'.format(figure_fname))
        plt.savefig(figure_fname, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_power_spectrum_two_layers(powers_negative, powers_positive, times, title='', figure_fname='',
                                   only_power_spectrum=True, high_gamma_max=120):
    if not only_power_spectrum:
        fig, (ax1, ax2, ax3) = plt.subplots(3)#, sharex=True)
    else:
        fig, ax1 = plt.subplots()
    F, T = powers_negative.shape
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55],
                 high_gamma=[65, high_gamma_max])

    im1 = _plot_powers(powers_negative, ax1, times, high_gamma_max)
    # cba = plt.colorbar(im1, shrink=0.25)
    im2 = _plot_powers(powers_positive, ax1, times, high_gamma_max)
    cbb = plt.colorbar(im2, shrink=0.25)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)

    if not only_power_spectrum:
        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_negative[idx, :], axis=0)
            ax2.plot(band_power.T, label=band_name)
        ax2.set_xlim([0, T])
        # ax2.legend()

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_positive[idx, :], axis=0)
            ax3.plot(band_power.T, label=band_name)
        ax3.set_xlim([0, T])
        # ax3.legend()

    if figure_fname != '':
        print('Saving figure to {}'.format(figure_fname))
        plt.savefig(figure_fname, dpi=300)
        plt.close()
    else:
        plt.show()


def _plot_powers(powers, ax, xaxis=None, high_gamma_max=120):
    from src.utils import color_maps_utils as cmu
    BuPu_YlOrRd_cm = cmu.create_BuPu_YlOrRd_cm()

    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    times = np.arange(powers.shape[1]) if xaxis is None else xaxis
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    if powers.shape[0] != len(freqs):
        print('powers.shape[0] != len(freqs)!!!')
        return

    if isinstance(powers, np.ndarray):
        vmax, vmin = np.max(powers), np.min(powers)
    else:
        vmax, vmin = np.ma.masked_array.max(powers), np.ma.masked_array.min(powers)
    if vmin > 0:
        cmap = matplotlib.cm.YlOrRd
        # cmap = 'YlOrRd'
    elif vmax < 0:
        cmap = matplotlib.cm.BuPu
        # cmap = 'BuPu'
    else:
        cmap = BuPu_YlOrRd_cm
        maxmin = max(map(abs, [vmax, vmin]))
        vmin, vmax = -maxmin, maxmin
    # powers[np.where(powers == 0)] = np.nan
    # cmap.set_bad(color='white')
    im = ax.imshow(np.flip(powers, 0), vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest',
               extent=extents(times) + extents(freqs), cmap=cmap)
    return im


def calc_masked_negative_and_positive_powers(norm_powers_min, norm_powers_max, percentiles=[5, 95]):
    minmax_powers = np.zeros(norm_powers_max.shape)
    min_inds = np.where(norm_powers_min < np.percentile(norm_powers_min, percentiles[0]))
    max_inds = np.where(norm_powers_max > np.percentile(norm_powers_max, percentiles[1]))
    minmax_powers[max_inds] = norm_powers_max[max_inds]
    minmax_powers[min_inds] = norm_powers_min[min_inds]
    negative_powers = masked_array(minmax_powers, minmax_powers < np.percentile(norm_powers_min, percentiles[0]))
    positive_powers = masked_array(minmax_powers, minmax_powers > np.percentile(norm_powers_max, percentiles[1]))
    return negative_powers, positive_powers


def plot_modalities_power_spectrums_with_graph(
        subject, modalities, window_fname, figure_name='', percentiles=[5, 95], inverse_method='dSPM', ylims=[-18, 6],
        file_type='jpg', cb_ticks = [], cb_ticks_font_size=12, figure_fol=''):

    evoked = mne.read_evokeds(window_fname)[0]
    times = evoked.times if len(evoked.times) % 2 == 0 else evoked.times[:-1]
    times = utils.downsample(times, 2)
    min_t, max_t = round(times[0]), round(times[-1])
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, 125, 5)])
    # bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    bands = dict(delta=[1, 4], high_gamma=[65, 120])
    if figure_fol == '':
        figure_fol = op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum')

    fig, ax = plt.subplots(3, len(modalities), figsize=(20, 10))
    for ind, modality in enumerate(modalities):
        powers_ax = ax[0, ind]
        powers_ax.set_title(modality.upper(), fontdict={'fontsize': 18})
        positive_powers_ax = ax[1, ind]
        negative_powers_ax = ax[2, ind]
        root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
            subject, inverse_method, modality, '{window}'))
        window = utils.namebase(window_fname)
        window_output_fname = output_fname.format(window=window)
        d = np.load(window_output_fname)
        powers_negative, powers_positive = calc_masked_negative_and_positive_powers(d['min'], d['max'], percentiles)

        im1 = _plot_powers(powers_negative, powers_ax, times)
        im2 = _plot_powers(powers_positive, powers_ax, times)
        if ind == 0:
            powers_ax.set_ylabel('Frequency (Hz)')
        else:
            powers_ax.set_yticks([])
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(powers_ax, width="5%", height="100%", loc=5,
                               bbox_to_anchor=(1.15, 0, 1, 1), bbox_transform=powers_ax.transAxes)
            cb = plt.colorbar(im2, cax=axins)
            if cb_ticks != []:
                cb.set_ticks(cb_ticks)
            cb.ax.tick_params(labelsize=cb_ticks_font_size)
            cb.ax.set_ylabel('dBHZ Z-Score', color='black', fontsize=cb_ticks_font_size)

        # for powers, axis, positive in zip([powers_positive, powers_negative], ax[1:2, ind], [True, False]):
        #     for band_name, band_freqs in bands.items():
        #         idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        #         band_power = np.mean(powers[idx, :], axis=0)
        #         band_power[abs(band_power) < 2] = 0
        #         axis.plot(times, band_power.T, label=band_name)
        #     axis.set_xlim([min_t, max_t])
        #     axis.set_ylim([2, ylims[1]] if positive else [ylims[0], -2])
        #     if ind == 0:
        #         axis.set_ylabel('Positive Z-Scores' if positive else 'Negative Z-Scores')
        #     else:
        #         axis.set_yticks([])
        #         axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # negative_powers_ax.set_xlabel('Time (s)')

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_negative[idx, :], axis=0)
            band_power[abs(band_power) < 2] = 0
            positive_powers_ax.plot(times, band_power.T, label=band_name.replace('_', ' '))
        positive_powers_ax.set_xlim([min_t, max_t])
        positive_powers_ax.set_ylim([2, ylims[1]])
        if ind == 0:
            positive_powers_ax.set_ylabel('Positive Z-Scores')
        else:
            positive_powers_ax.set_yticks([])
        if ind == len(modalities) - 1:
            positive_powers_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_positive[idx, :], axis=0)
            band_power[abs(band_power) < 2] = 0
            negative_powers_ax.plot(times, band_power.T, label=band_name.replace('_', ' '))
        negative_powers_ax.set_xlim([min_t, max_t])
        negative_powers_ax.set_ylim([ylims[0], -2])
        if ind == 0:
            negative_powers_ax.set_ylabel('Negative Z-Scores')
        else:
            negative_powers_ax.set_yticks([])
        if ind == len(modalities) - 1:
            negative_powers_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        negative_powers_ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=0.92, top=None, wspace=None, hspace=None)
    if figure_name != '':
        plt.savefig(op.join(figure_fol, '{}.{}'.format(figure_name, file_type)), dpi=300)
        plt.close()
    else:
        plt.show()


def calc_induced_power_zvals(
        subject, windows_fnames, baseline_name, modality, bands, from_index=None, to_index=None, inverse_method='dSPM',
        parallel=True, overwrite=False):
    params = [(subject, modality, window_fname, baseline_name, bands, from_index, to_index, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_induced_power_zvals_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_induced_power_zvals_parallel(p):
    subject, modality, window_fname, baseline_name, bands, from_index, to_index, inverse_method, overwrite = p
    module = eeg if modality == 'eeg' else meg
    stc_template = '{}-epilepsy-{}-{}-{}_{}'.format(subject, inverse_method, modality, '{window}', '{band}')
    root_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    if all([utils.stc_exist(op.join(root_fol, '{}-zvals'.format(
            stc_template.format(window=utils.namebase(window_fname), band=band)))) \
            for band in bands]) and not overwrite:
        return
    for band in bands:
        window_stc_name = stc_template.format(window=utils.namebase(window_fname), band=band)
        args = module.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc_zvals',
            task='epilepsy',
            stc_name=window_stc_name,
            baseline_stc_name=stc_template.format(window=baseline_name, band=band),
            from_index=from_index,
            to_index=to_index,
            use_abs=1,
            overwrite_stc=overwrite
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
        op.join(MMVT_DIR, subject, 'meg', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name))) + \
        glob.glob(op.join(MMVT_DIR, subject, 'eeg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name)))
    if len(stc_fnames) == 0:
        stc_fnames = glob.glob(
            op.join(MMVT_DIR, subject, 'meg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name))) + \
            glob.glob(op.join(MMVT_DIR, subject, 'eeg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name)))
    if len(stc_fnames) == 0:
        print('No baselines stc files were found!')
        return
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'baseline'))
    utils.run_parallel(_plot_stcs_files_parallel, [(stc_fname, figures_fol) for stc_fname in stc_fnames], n_jobs)


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
        # if op.isfile(fig_fname) and not overwrite:
        #     print('{} already exist'.format(fig_fname))
        #     break
        plt.figure()
        all_files_found = True
        for modality in modalities:
            modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
            stc_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_amplitude-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name))
            if not op.isfile(stc_fname):
                print('Can\'t find {}!'.format(stc_fname))
                all_files_found = False
                break
            stc = mne.read_source_estimate(stc_fname)
            data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
            plt.plot(data.T)
        if all_files_found:
            plt.title('{} amplitude'.format(window_name))
            plt.legend(modalities)
            print('Saving {} amplitude'.format(window_name))
            plt.savefig(fig_fname, dpi=300)
            plt.close()


def plot_modalities(subject, windows, modalities, bands, inverse_method, max_t=0, overwrite=False, n_jobs=4):
    from itertools import product
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'modalities'))
    params = [(subject, modalities, inverse_method, figures_fol, band, window_fname, max_t, overwrite)
              for (band, window_fname) in product(bands, windows)]
    utils.run_parallel(_plot_modalities_parallel, params, n_jobs)


def _plot_modalities_parallel(p):
    subject, modalities, inverse_method, figures_fol, band, window_fname, max_t, overwrite = p
    window_name = utils.namebase(window_fname)
    plt.figure()
    for modality in modalities:
        modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
        fig_fname = op.join(figures_fol, '{}-{}.jpg'.format(window_name, band))
        if op.isfile(fig_fname) and not overwrite:
            print('{} already exist'.format(fig_fname))
            break
        stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
            subject, inverse_method, modality, window_name, band))
        if not op.isfile(stc_band_fname):
            print('Can\'t find {}!'.format(stc_band_fname))
            break
        print('Loading {} ({})'.format(stc_band_fname, utils.file_modification_time(stc_band_fname)))
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


def add_run_number_to_files(subject, run):
    run_num = re.sub('\D', ',', run).split(',')[-1]
    files = glob.glob(op.join(MEG_DIR, subject, '{}-epilepsy-*.fif'.format(subject))) + \
            glob.glob(op.join(EEG_DIR, subject, '{}-epilepsy-*.fif'.format(subject)))
            # glob.glob(op.join(MMVT_DIR, subject, 'meg', '{}-epilepsy-*.fif'.format(subject))) + \
            # glob.glob(op.join(MMVT_DIR, subject, 'eeg', '{}-epilepsy-*.fif'.format(subject)))
    for fname in files:
        new_fname = fname.replace('-epilepsy-', '-epilepsy{}-'.format(run_num))
        print('{} - > {}'.format(fname, new_fname))
        utils.rename_files(fname, new_fname)


def create_evokeds_links(subject, windows):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'evoked'))
    for window_fname in windows:
        new_window_fname = op.join(fol, utils.namebase_with_ext(window_fname))
        if op.isfile(new_window_fname) or op.islink(new_window_fname):
            continue
        utils.make_link(window_fname, new_window_fname)


def plot_evokes(subject, modality, windows, bad_channels, parallel=True, overwrite=False):
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'evokes'))
    params = [(subject, modality, window_fname, bad_channels, figs_fol, overwrite) for window_fname in windows]
    utils.run_parallel(_plot_evokes_parallel, params, len(windows) if parallel else 1)


def _plot_evokes_parallel(p):
    subject, modality, window_fname, bad_channels, figs_fol, overwrite = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    evo_fname = op.join(MMVT_DIR, subject, 'evoked', '{}.fif'.format(window))
    if not op.isfile(evo_fname):
        utils.make_link(window_fname, evo_fname)
    fig_fname = op.join(figs_fol, '{}.jpg'.format(window))
    if op.isfile(fig_fname) and not overwrite:
        return
    if bad_channels != 'bads':
        bad_channels = bad_channels.split(',')
    module.plot_evoked(
        subject, evo_fname, window_title=window, exclude=bad_channels, save_fig=True,
        fig_fname=fig_fname, overwrite=overwrite)


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
    if bad_channels != 'bads':
        bad_channels = bad_channels.split(',')
    module.plot_topomap(
        subject, evo_fname, times=[0], find_peaks=True, same_peaks=False, n_peaks=5, bad_channels=bad_channels,
        title=window, save_fig=True, fig_fname=fig_fname)


def find_vertices(subject, modality='meg', atlas='aparc.DKTatlas'):
    from src.utils import labels_utils as lu
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    vertices = []
    for ind, label in enumerate(labels):
        _, vertno = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        vertices.extend(vertno)
    np.save(op.join(MMVT_DIR, subject, 'labels_verts.npy'), np.array(vertices))
    return vertices


def calc_stc_power_specturm(subject, modality, window_fname, baseline_window, inverse_method='dSPM', atlas='aparc.DKTatlas'):
    from src.utils import labels_utils as lu
    from collections import defaultdict
    import time

    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    window = utils.namebase(window_fname)
    output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
        subject, inverse_method, modality, '{window}'))
    window_output_fname = output_fname.format(window=window)

    d = np.load(window_output_fname)
    min_indeices, max_indices = d['min_vertices'], d['max_vertices']
    x = d['max'][:5, 4000:4050]
    max_val = np.max(x)
    f, t = np.unravel_index(x.argmax(), x.shape)
    t += 4000
    print('{} in {}f {}t'.format(max_val, f, t))
    vertices_ind = max_indices[f, t]

    # this_vertno = np.intersect1d(vertno, label.vertices)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)

    labels_norm_data_fol = utils.make_dir(op.join(root_dir, 'labels_norm_all_baseline'))
    baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(baseline_window)))
    baseline_files = glob.glob(op.join(baseline_fol, 'epilepsy_*_induced_power.npy'))
    powers_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, window))
    powers_files = glob.glob(op.join(powers_fol, 'epilepsy_*_induced_power.npy'))
    all_files_exist = True
    for powers_fname in powers_files:
        label_name = utils.namebase(powers_fname).split('_')[1]
        label_fname = op.join(labels_norm_data_fol, '{}-epilepsy-{}-{}-{}-{}-norm-induced_power.npy'.format(
            subject, inverse_method, modality, window, label_name))
        all_files_exist = all_files_exist and op.isfile(label_fname)

    if not all_files_exist:
        baseline = concatenate_powers(baseline_fol) # (vertices x freqs x time)
        baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the mean (over time) of log baseline values
        powers = concatenate_powers(powers_fol)
        norm_powers = (powers - baseline_mean) / baseline_std

    # norm_labels_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
    #     subject, inverse_method, modality, window))
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    vertices, vertices_data = defaultdict(list), defaultdict(list)
    # labels_data = []
    start_ind = 0
    now = time.time()
    for file_ind, (baseline_fname, powers_fname) in enumerate(zip(baseline_files, powers_files)):
        if not all_files_exist:
            utils.time_to_go(now, file_ind, len(baseline_files), 1)
        baseline_label_name = utils.namebase(baseline_fname).split('_')[1]
        powers_label_name = utils.namebase(powers_fname).split('_')[1]
        if baseline_label_name != powers_label_name:
            raise Exception('ASDGF@#Q%EGF#Q$T')
        label_name = powers_label_name
        label = [l for l in labels if l.name == label_name][0]
        label_fname = op.join(labels_norm_data_fol, '{}-epilepsy-{}-{}-{}-{}-norm-induced_power.npy'.format(
            subject, inverse_method, modality, window, label_name))
        vertno, src_sel = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        if not op.isfile(label_fname):
            print('Reading norm_powers ({}) from {}:{}'.format(norm_powers.shape[0], start_ind, start_ind+len(src_sel)))
            label_powers = norm_powers[start_ind: start_ind+len(src_sel)]
            np.save(label_fname, label_powers)
        else:
            label_powers = np.load(label_fname, mmap_mode='r')
        start_ind += len(src_sel)
        labels_powers_per_freq_per_time = label_powers[:, f, t]
        # if vertices_ind in label.vertices:
        if np.isclose(max_val, np.max(labels_powers_per_freq_per_time)):
            print('label {}, for {}f and {}t: {}'.format(label_name, f, t, np.max(labels_powers_per_freq_per_time)))
        vertices[label.hemi].extend(vertno[0] if label.hemi == 'lh' else vertno[1])
        vertices_data[label.hemi].extend(labels_powers_per_freq_per_time)
        del label_powers

    evoked = mne.read_evokeds(window_fname)[0]
    combined_stc = meg.creating_stc_obj(
        vertices_data, vertices, subject, tmin=evoked.times[0], tstep=evoked.times[1] - evoked.times[0])
    output_stc_fname = op.join(
        MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg',
        'epilepsy-{}-{}-{}-{}t-{}f-induced_norm_power'.format(inverse_method, modality, window, t, f))
    print('Saving stc file for t {} and f {} in {}'.format(t, f, output_stc_fname))
    combined_stc.save(output_stc_fname)


    # label_baseline = np.load(op.join(baseline_fol, 'epilepsy_{}_induced_power.npy'.format(max_label.name))).astype(np.float32)
    # # label_baseline_std = np.std(label_baseline, axis=2, keepdims=True)  # the standard deviation (over time) of log baseline values
    # # label_baseline_mean = np.mean(label_baseline, axis=2, keepdims=True)  # the mean (over time) of log baseline values
    #
    # label_baseline_std = np.std(label_baseline * pow(10, -15), axis=2, keepdims=True) * pow(10, 15)
    # label_baseline_mean = np.mean(label_baseline * pow(10, -15), axis=2, keepdims=True) * pow(10, 15)
    #
    #

    # label_powers = np.load(op.join(powers_fol, 'epilepsy_{}_induced_power.npy'.format(max_label.name))).astype(np.float32)
    # norm_powers = (label_powers - label_baseline_mean) / label_baseline_std

    print('sadf')


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_template,
         inverse_method='dSPM', no_runs=False, n_jobs=4):
    run_num = re.sub('\D', ',', run).split(',')[-1]
    if no_runs:
        windows = glob.glob(op.join(evokes_fol, '*.fif'))
        baseline_windows = glob.glob(op.join(evokes_fol, '*{}*.fif'.format(baseline_template)))
    else:
        windows = glob.glob(op.join(evokes_fol, '{}_*.fif'.format(run)))
        baseline_windows = glob.glob(op.join(evokes_fol, '{}*.fif'.format(run, baseline_template)))
    for baseline_window in baseline_windows:
        windows.remove(baseline_window)
    windows_with_baseline = windows + baseline_windows
    baseline_name = utils.namebase(baseline_windows[0])
    overwrite_inv = False
    overwrite_fwd = False
    overwrite_evokes = True
    check_for_labels_files = False
    overwrite_induced_power_zvals = False
    overwrite_stc = False
    overwrite_modalities_figures = False
    from_index, to_index = 2000, 10000
    max_t = 0 #7500
    high_gamma_max = 300
    percentiles = [50, 50] # [5, 95]

    # create_evokeds_links(subject, windows_with_baseline)
    for modality in modalities:
        # 1) Sensors
        # plot_evokes(subject, modality, windows, bad_channels, n_jobs > 1, overwrite_evokes)
        # plot_topomaps(subject, modality, windows, bad_channels, parallel=n_jobs > 1)

        # calc_sensors_power(subject, windows_with_baseline, modality, inverse_method, bad_channels,
        #                    high_gamma_max=high_gamma_max, downsample=2, parallel=n_jobs > 1, overwrite=True)
        # plot_sensors_powers(subject, windows, baseline_window, modality, inverse_method,
        #                     high_gamma_max=high_gamma_max, percentiles=percentiles,
        #                     overwrite=True, parallel=False)

        # 2) calc fwd and inv
        calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels,
                     overwrite_inv=overwrite_inv, overwrite_fwd=overwrite_fwd)
        check_inv_fwd(subject, modality, run_num)

        # 3) Amplitude
        # calc_amplitude(subject, modality, run_num, windows_with_baseline, inverse_method, overwrite_stc, n_jobs)
        # calc_amplitude_zvals(
        #     subject, windows, baseline_name, modality, from_index, to_index, inverse_method,
        #     parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)

        # 4) Induced power
        # calc_induced_power(subject, run_num, windows_with_baseline, modality, inverse_method, check_for_labels_files,
        #                    overwrite=True)
        # plot_norm_powers(subject, windows, baseline_window, modality, inverse_method, use_norm_labels_powers=False,
        #                  overwrite=False, figures_type='eps')
        # plot_norm_powers_per_label(subject, windows, baseline_window, modality, inverse_method,
        #                            calc_also_non_norm_powers=False, overwrite=True, n_jobs=n_jobs)
        # calc_stc_power_specturm(subject, modality, windows[0], baseline_window)
        pass

    # find_vertices(subject)
    # for window_fname in windows:
    #     figure_name = '' #'{}-modalities-power-spectrum-with-grpahs'.format(utils.namebase(window_fname))
    #     plot_modalities_power_spectrums_with_graph(subject, modalities, window_fname, figure_name, file_type='eps')

    # files = glob.glob('/autofs/space/thibault_001/users/npeled/EEG/nmr01321/nmr01321-epilepsy-dSPM-eeg-run1_szMEG_213s-induced_power/epilepsy_*_induced_norm_power.npy')
    # calc_powers_abs_minmax(None, label_norm_powers_files=files)

    # Old stuff
    # for modality in modalities:
        # calc_max_powers(subject, windows_with_baseline, modality, inverse_method, overwrite=False, parallel=True)
        # plot_max_powers(subject, windows_with_baseline, modality, inverse_method, overwrite=False, parallel=False)
        # calc_induced_power_zvals(
        #     subject, windows, baseline_name, modality, bands, from_index, to_index, inverse_method,
        #     parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)
        # move_non_zvals_stcs(subject, modality)
        # plot_stcs_files(subject, modality, n_jobs)
        # plot_windows(subject, windows, modality, bands, inverse_method)
        # plot_freqs(subject, temporal_windows, modality, bands, inverse_method, max_t)

    # plot_modalities(subject, windows, modalities, bands, inverse_method, max_t, overwrite_modalities_figures, n_jobs)
    # plot_activity_modalities(subject, windows, modalities, inverse_method, overwrite=overwrite_modalities_figures)
    # plot_baseline(subject, baseline_name)
    # fix_amplitude_fnames(subject, bands)


def subject_nmr00857():
    subject = 'nmr00857'
    evokes_fol = op.join(MEG_DIR, subject, 'evoked')
    meg_fol = '/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123'
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG061,EEG02,EEG042,MEG0112,MEG0113'
    baseline_name = 'BaseLINE'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name


def subject_nmr01321():
    subject = 'nmr01321'
    evokes_fol = [d for d in [
        # '/autofs/space/frieda_001/users/valia/epilepsy/4272326_01321/MMVT_epochs',
        '/homes/5/npeled/space1/MEG/nmr01321/evokeds',
        op.join(MMVT_DIR, subject, 'evoked')] if op.isdir(d)][0]
    meg_fol = [d for d in [
        '/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123',
        op.join(MEG_DIR, subject)] if op.isdir(d)][0]
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG001,EEG003,EEG004,EEG005,EEG008,EEG034,EEG045,EEG051,EEG057,EEG058,EEG060,EEG061,EEG062,EEG074,MEG1422,MEG1532,MEG2012,MEG2022'
    baseline_name = 'Base_line'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name


def subject_nmr01325():
    subject = 'nmr01325'
    evokes_fol = [d for d in [
        '/cluster/neuromind/valia/epilepsy/6645962_01325',
        op.join(MEG_DIR, subject, 'evokes')] if op.isdir(d)][0]
    meg_fol = [d for d in [
        '/cluster/neuromind/valia/epilepsy/6645962_01325/190523',
        op.join(MEG_DIR, subject)] if op.isdir(d)][0]
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG020,EEG021,EEG050,EEG051'
    baseline_name = 'baseline'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name


def find_room_noise(fol):
    files = glob.glob(op.join(fol, '*roomnoise_raw.fif'))
    if len(files) >= 1:
        return utils.select_one_file(files, 'room noise')
    files = glob.glob(op.join(fol, '*noiseroom_raw.fif'))
    if len(files) >= 1:
        return utils.select_one_file(files, 'room noise')
    file_name = input('Can\'t find the room noise file, please input its name: ')
    fname = op.join(fol, file_name)
    if not op.isfile(fname):
        raise Exception('No room noise!')
    else:
        return fname


if __name__ == '__main__':
    modalities = ['eeg', 'meg', 'meeg']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    inverse_method = 'dSPM'
    subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name = subject_nmr01325()
    no_runs = False
    runs = set([utils.namebase(f).split('_')[0] for f in glob.glob(op.join(evokes_fol, 'run*_*.fif'))])
    if len(runs) == 0:
        print('No run were found!')
        runs = ['01']
        no_runs = True
    n_jobs = 1 # utils.get_n_jobs(-5)
    print('n_jobs: {}'.format(n_jobs))
    for run in runs:
        # if run != 'run1':
        #     continue
        if len(runs) > 0:
            raw_fname = ''
            run_num = re.sub('\D', ',', run).split(',')[-1]
            raw_files = glob.glob(op.join(meg_fol, '*_{}_*raw*.fif'.format(str(run_num).zfill(2))))
            ssst_raw_files = [f for f in raw_files if 'ssst' in utils.namebase(f)]
            if not op.isfile(raw_fname):
                raw_run_files = [f for f in raw_files if 'annot' not in utils.namebase(f) and 'eve' not in utils.namebase(f)]
                if len(raw_run_files) == 0:
                    continue
                ssst_raw_files = [f for f in raw_run_files if 'ssst' in utils.namebase(f)]
                if len(ssst_raw_files) > 0:
                    raw_fname = utils.select_one_file(ssst_raw_files, 'raw file for run {}'.format(run_num))
                else:
                    raw_fname = utils.select_one_file(raw_run_files, 'raw file for run {}'.format(run_num))
                if not op.isfile(raw_fname):
                    continue
        main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_name,
             inverse_method, no_runs, n_jobs)
    print('Finish!')