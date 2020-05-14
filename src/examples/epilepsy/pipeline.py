from builtins import enumerate

from src.preproc import eeg
from src.preproc import meg
from src.preproc import connectivity
from src.utils import utils
from src.utils import labels_utils as lu
import glob
import os.path as op
import mne
import numpy as np
import os
from src.examples.epilepsy import utils as epi_utils
from src.examples.epilepsy import plots
from src.examples.epilepsy import power_spectrums_plots as psplots

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


def calc_amplitude(subject, modality, windows_fnames, inverse_method='dSPM', downsample_r=1, overwrite=False,
                   rename=True, n_jobs=4):
    params = [(subject, window_fname, modality, windows_fnames, inverse_method, downsample_r, overwrite, rename)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_amplitude_parallel, params, n_jobs)


def _calc_amplitude_parallel(p):
    # python3 -m src.preproc.meg -s nmr00857 -f calc_stc -i dSPM -t epilepsy
    #   --evo_fname /autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/43.9s.fif
    #   --overwrite_stc 1
    subject, window_fname, modality, windows_fnames, inverse_method, downsample_r, overwrite, rename = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    stc_name = op.join(MMVT_DIR, subject, modality, '{}-epilepsy-{}-{}-{}'.format(
            subject, inverse_method, modality, utils.namebase(window_fname)))
    if not utils.stc_exist(stc_name) or overwrite:
        args = module.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            task='epilepsy',
            inverse_method=inverse_method,
            inv_fname=op.join(root_dir, '{}-epilepsy-{}-inv.fif'.format(subject, modality)),
            fwd_fname=op.join(root_dir, '{}-epilepsy-{}-fwd.fif'.format(subject, modality)),
            fwd_usingEEG=modality in ['eeg', 'meeg'],
            evo_fname=window_fname,
            stc_template=stc_name,
            downsample_r=downsample_r,
            overwrite_stc=overwrite,
            n_jobs=1))
        module.call_main(args)
    else:
        print('{}-?h.stc already exist'.format(utils.namebase(stc_name)))
    if rename:
        move_and_rename_amplitude_files(subject, modality, inverse_method)


def move_and_rename_amplitude_files(subject, modality, inverse_method='dSPM'):
    stcs_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    zvals_fol = utils.make_dir(op.join(stcs_fol, 'zvals'))
    no_zvals_fol = utils.make_dir(op.join(stcs_fol, 'no-zvals'))
    stcs_files = glob.glob(op.join(stcs_fol, '{}-epilepsy-{}-{}*-?h.stc'.format(subject, inverse_method, modality)))
    for stc_fname in stcs_files:
        stc_name = utils.namebase(stc_fname)[:-3]
        if stc_name.endswith('zvals'):
            utils.move_file(stc_fname, zvals_fol, overwrite=True)
        else:
            new_stc_fname = '{}-amplitude-{}'.format(stc_fname[:-len('-rh.stc')], stc_fname[-len('rh.stc'):])
            os.rename(stc_fname, new_stc_fname)
            try:
                utils.move_file(new_stc_fname, no_zvals_fol, overwrite=True)
            except:
                print('move_and_rename_amplitude_files: Error in renaming!')
                print('{} -> {}'.format(new_stc_fname, no_zvals_fol))


def calc_amplitude_zvals(subject, windows_fnames, baseline_name, modality, from_index=None, to_index=None,
                         inverse_method='dSPM', use_abs=False, parallel=True, overwrite=False):
    params = [(subject, modality, window_fname, baseline_name, from_index, to_index, inverse_method, use_abs,
               overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_amplitude_zvals_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_amplitude_zvals_parallel(p):
    # python3 -m src.preproc.meg -s nmr00857 -f calc_stc_zvals --stc_name nmr00857-epilepsy-dSPM-meeg-43.9s
    #   --baseline_stc_name nmr00857-epilepsy-dSPM-meeg-37.3_BGprSzs --use_abs 1 --overwrite_stc 1
    subject, modality, window_fname, baseline_name, from_index, to_index, inverse_method, use_abs, overwrite = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    stc_template = '{}-epilepsy-{}-{}-{}{}'.format(subject, inverse_method, modality, '{window}', '{suffix}')
    output_name = stc_template.format(window=window, suffix='_amplitude-zvals')
    output_fname = find_stc_file(output_name, modality)
    if utils.stc_exist(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
        return
    window_stc_name = stc_template.format(window=window, suffix='')
    baseline_stc_name = stc_template.format(window=baseline_name, suffix='')
    stc_fname = find_stc_file(window_stc_name, modality)
    baseline_fname = find_stc_file(baseline_stc_name, modality)
    if stc_fname == '' or baseline_fname == '':
        return
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_stc_zvals',
        task='epilepsy',
        stc_name=stc_fname,
        baseline_stc_name=baseline_fname,
        stc_zvals_name=stc_template.format(window=window, suffix='_amplitude-zvals'),
        from_index=from_index,
        to_index=to_index,
        use_abs=use_abs,
        overwrite_stc=overwrite
    ))
    module.call_main(args)
    move_and_rename_amplitude_files(subject, modality, inverse_method)


def find_stc_file(window_stc_name, modality):
    stcs_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    if 'amplitude-zvals' in window_stc_name:
        stcs = glob.glob(op.join(stcs_fol, 'zvals', '{}-?h.stc'.format(window_stc_name)))
    else:
        stcs = glob.glob(op.join(stcs_fol, '**', '{}-*?h.stc'.format(window_stc_name)), recursive=True)
    if len(stcs) == 2 and stcs[0][:-7] == stcs[1][:-7]:
        window_stc_name = stcs[0][:-7]
    else:
        print('Can\'t find stc files for {}'.format(window_stc_name))
        if len(stcs) > 0:
            for stc_fname in stcs:
                print(stc_fname)
        window_stc_name = ''
    return window_stc_name


def average_amplitude_zvals(subject, windows, modality, output_name, use_abs=False, inverse_method='dSPM',
                            do_plot=False, overwrite=False):
    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure()
    root_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'average_amplitude_zvals'))
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-{}-average-amplitude-zvals.jpg'.format(
            subject, inverse_method, modality, output_name, '{window}'))
    output_fname = op.join(root_fol, '{}-epilepsy-{}-{}-{}-average-amplitude-zvals'.format(
        subject, inverse_method, modality, output_name))
    if utils.stc_exist(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
        return
    times, vertices = None, {}
    data = {'rh': [], 'lh': []}
    for window_fname in windows:
        window_name = utils.namebase(window_fname)
        stc_template = op.join(root_fol, 'zvals', '{}-epilepsy-{}-{}-{}_amplitude-zvals'.format(
            subject, inverse_method, modality, window_name))
        if not utils.stc_exist(stc_template):
            print('Can\'t find {}!'.format(stc_template))
            continue
        if times is None:
            times = epi_utils.get_window_times(window_fname, downsample=1)
        print('Reading {}'.format('{}-rh.stc'.format(stc_template)))
        stc = mne.read_source_estimate('{}-rh.stc'.format(stc_template), subject)
        data['rh'].append(stc.rh_data)
        data['lh'].append(stc.lh_data)
        if len(vertices) == 0:
            vertices['rh'] = stc.rh_vertno
            vertices['lh'] = stc.lh_vertno
        if do_plot:
            max_data = np.max(np.abs(stc.data), axis=0)
            plt.plot(times, max_data.T, label=window_name)
    if do_plot:
        plt.legend()
        plt.title('{} {}: all windows max'.format(modality, inverse_method))
        print('Saving {}'.format(figures_template.format(window='all-max')))
        plt.savefig(figures_template.format(window='all-max'))
        plt.close()
    for hemi in utils.HEMIS:
        if use_abs:
            data[hemi] = np.abs(np.array(data[hemi])).mean(0)
        else:
            data[hemi] = np.array(data[hemi]).mean(0)
    avg_stc = meg.creating_stc_obj(
        data, vertices, subject, tmin=times[0], tstep=times[1] - times[0])
    if do_plot:
        plt.figure()
        max_data = np.max(np.abs(avg_stc.data), axis=0)
        plt.plot(times, max_data.T)
        plt.title('{} {}: avg stc'.format(modality, inverse_method))
        print('Saving {}'.format(figures_template.format(window='avg')))
        plt.savefig(figures_template.format(window='avg'))
        plt.close()
    print('Saveing avg stc in {}'.format(output_fname))
    avg_stc.save(output_fname)


def calc_sensors_power(subject, windows_fnames, modality, inverse_method='dSPM', bad_channels=[],
                       high_gamma_max=120, downsample=2, parallel=False, overwrite=False):
    params = [(subject, window_fname, modality, inverse_method, bad_channels, downsample, high_gamma_max, overwrite)
              for window_fname in windows_fnames if op.isfile(window_fname)]
    utils.run_parallel(_calc_sensors_power_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_sensors_power_parallel(p):
    from mne.time_frequency import tfr_array_morlet

    subject, window_fname, modality, inverse_method, bad_channels, downsample, high_gamma_max, overwrite = p

    root_dir = utils.make_dir(op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject))
    output_fname_template = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_power.npy'.format(
        subject, inverse_method, modality, '{window}'))

    window = utils.namebase(window_fname)
    output_fname = output_fname_template.format(window=window)
    if op.isfile(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
        return
    evoked = mne.read_evokeds(window_fname)[0]
    # evoked = evoked.resample(1000)

    if modality == 'eeg':
        picks = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=bad_channels)
    elif modality == 'meg':
        picks = mne.pick_types(evoked.info, meg=True, eeg=False, exclude=bad_channels)
    elif modality == 'meeg':
        picks = mne.pick_types(evoked.info, meg=True, eeg=True, exclude=bad_channels)
    else:
        raise Exception('Wrong modality!')

    evoked_data = evoked.data[np.newaxis, picks, :]
    freqs, n_cycles = calc_morlet_freqs(evoked, high_gamma_max)
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


def calc_morlet_freqs(evoked, high_gamma_max=120):
    from mne.time_frequency import morlet
    import math

    T = evoked.data.shape[1]
    sfreq = evoked.info['sfreq']
    low_freq = 1
    if T / sfreq < 5 * low_freq:
        low_freq = math.ceil((sfreq * 5) / T)
    freqs = epi_utils.get_freqs(low_freq, high_gamma_max)
    # n_cycles = freqs / 2.

    n_cyles_div = 2.
    f0_n_cycles = freqs[0] / n_cyles_div
    sigma_t = f0_n_cycles / (2.0 * np.pi * freqs[0])
    t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
    t = np.r_[-t[::-1], t[1:]]
    while len(t) > T:
        n_cyles_div += 1.
        f0_n_cycles = freqs[0] / n_cyles_div
        sigma_t = f0_n_cycles / (2.0 * np.pi * freqs[0])
        t = np.arange(0., 5. * sigma_t, 1.0 / evoked.info['sfreq'])
        t = np.r_[-t[::-1], t[1:]]

    n_cycles = freqs / n_cyles_div
    W = morlet(evoked.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)
    while len(W[0]) > T:
        low_freq += 1
        freqs = epi_utils.get_freqs(low_freq, high_gamma_max)
        n_cycles = freqs / 2.
        W = morlet(evoked.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)

    return freqs, n_cycles


def calc_induced_power(subject, run_num, windows_fnames, modality, inverse_method='dSPM', check_for_labels_files=True,
                       overwrite=False):

    def files_exist(window_fname):
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, utils.namebase(window_fname)))
        if not op.isdir(fol):
            return False
        files = glob.glob(op.join(fol, 'epilepsy_*_induced_power.npy'))
        if len(files) < 62:
            return False
        for fname in files:
            # file_mod_time = utils.file_modification_time_struct(fname)
            # if not (file_mod_time.tm_year >= 2019 and (file_mod_time.tm_mon == 7 and file_mod_time.tm_mday >= 10) or \
            #         (file_mod_time.tm_mon > 7)):
            if not utils.file_mod_after_date(fname, 10, 7, 2019):
                return False
        return True

    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    files_to_calc = [utils.namebase(window_fname) for window_fname in windows_fnames if not files_exist(window_fname)]
    if len(files_to_calc) == 0 and not overwrite:
        print('All files exist!')
        return
    else:
        print('Files needed to recalc:')
        for ind, fname in enumerate(files_to_calc):
            print('{}: {}'.format(ind + 1, fname))
        # ret = input('Do you want to continue (y/n)? ')
        # if not au.is_true(ret):
        #     return
    # output_fname = op.join(MMVT_DIR, 'eeg' if modality == 'eeg' else 'meg', '{}-epilepsy-{}-{}-{}_{}'.format(
    #     subject, inverse_method, modality, '{window}', '{band}'))
    for window_fname in windows_fnames:
        print('{} {} {}:'.format(subject, modality, utils.namebase(window_fname)))
        if files_exist(window_fname) and not overwrite:
            print('Already exist')
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
    max_powers = np.max(epi_utils.concatenate_powers(fol), axis=0)
    print('Saving to {}'.format(output_fname))
    np.save(output_fname, max_powers)


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


def calc_stc_power_specturm(subject, modality, power_stc_name, window_fname, baseline_window, avg_time_crop, run_num,
                            inverse_method='dSPM', atlas='aparc.DKTatlas', high_gamma_max=120, win_suffix='',
                            vertices_ind=None):
    from collections import defaultdict
    import time

    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    window = utils.namebase(window_fname)
    output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
        subject, inverse_method, modality, '{window}'))
    window_output_fname = output_fname.format(window='{}{}'.format(power_stc_name, win_suffix))
    if not op.isfile(window_output_fname):
        print('Can\'t find {}!'.format(window_output_fname))
        return

    d = np.load(window_output_fname)
    freqs = epi_utils.get_freqs(d['min_f_ind'] + 1, high_gamma_max)
    times = epi_utils.get_window_times(window_fname, downsample=2)
    # t_max = np.where(times > 0.1)[0][0]
    t_min = 0 #np.where(times > 0)[0][0]
    x = np.flip(d['max'], 0) #[:, t_min:t_max]
    max_val = np.max(x)
    f, t = np.unravel_index(x.argmax(), x.shape)
    t += t_min + avg_time_crop
    f -= d['min_f_ind']
    print('norm_powers_maxs: {:.3f} at {:.2f}s and {}Hz'.format(max_val, times[t], freqs[f]))
    # min_indices, max_indices = d['min_vertices'], d['max_vertices']
    # vertices_ind = max_indices[max_f, max_t]

    # max_f, max_t = np.unravel_index(np.flip(x, 0).argmax(), x.shape)


    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    # max_vert_label = find_vert_label(vertices_ind, modality, window, labels, inv)
    # vertices_ind_to_no_lookup, vertices_labels_lookup = \
    #     epi_utils.calc_vertices_lookup_tables(subject, modality, window, inverse_method, labels, inv)
    # max_vert_label = vertices_labels_lookup[vertices_ind]
    # print('{:.4f} in {}Hz {:.4f}s in {}'.format(max_val, freqs[f], times[t], max_vert_label.name))

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
        baseline = epi_utils.concatenate_powers(baseline_fol) # (vertices x freqs x time)
        baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the mean (over time) of log baseline values
        powers = epi_utils.concatenate_powers(powers_fol)
        norm_powers = (powers - baseline_mean) / baseline_std

    # norm_labels_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
    #     subject, inverse_method, modality, window))
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
            print('Reading norm_powers ({}) from {}:{} for label {}'.format(
                norm_powers.shape[0], start_ind, start_ind+len(src_sel), label.name))
            label_powers = norm_powers[start_ind: start_ind+len(src_sel)]
            np.save(label_fname, label_powers)
        else:
            label_powers = np.load(label_fname, mmap_mode='r')
        labels_powers_per_freq_per_time = label_powers[:, f, t]
        if start_ind <= vertices_ind < start_ind+len(src_sel):
            print('vertices_ind: label {}, for {}Hz and {:.3f}s: {}'.format(
                label_name, freqs[f], times[t], label_powers[vertices_ind - start_ind, f, t]))
            if label_powers[vertices_ind - start_ind, f, t] != max_val:
                raise Exception('@#%$#$%@#$%@#$%@%$')
        if np.isclose(max_val, np.max(labels_powers_per_freq_per_time)):
            print('isclose: label {}, for {}Hz and {:.3f}s: {}'.format(
                label_name, freqs[f], times[t], np.max(labels_powers_per_freq_per_time)))
        start_ind += len(src_sel)
        vertices[label.hemi].extend(vertno[0] if label.hemi == 'lh' else vertno[1])
        vertices_data[label.hemi].extend(labels_powers_per_freq_per_time)
        del label_powers

    combined_stc = meg.creating_stc_obj(
        vertices_data, vertices, subject, tmin=times[0], tstep=times[1] - times[0])
    output_stc_fname = op.join(
        MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg',
        'epilepsy-{}-{}-{:.3f}s-{}Hz-{}-induced_norm_power'.format(
            modality, window, times[t], freqs[f], inverse_method))
    print('Saving stc file for t {} and f {} in {}'.format(t, f, output_stc_fname))
    combined_stc.save(output_stc_fname)


def calc_avg_power_specturm_stc(
        subject, modality, power_stc_name, windows, baseline_window, avg_time_crop, run_num,
        inverse_method='dSPM', atlas='aparc.DKTatlas40', high_gamma_max=120, overwrite=False):
    from collections import defaultdict
    import time

    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(baseline_window)))
    powers_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(windows[0])))
    stcs_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    output_stc_fname = op.join(
        stcs_fol, '{}-epilepsy-{}-{}-{}-avg-power-zvals'.format(subject, inverse_method, modality, power_stc_name))

    def load_avg_norm_powers(baselineF):
        avg_norm_powers_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-avg-induced_power.npy'.format(
            subject, inverse_method, modality, power_stc_name))
        powersF = 0

        if not op.isfile(avg_norm_powers_fname) or overwrite:
            avg_norm_powers = None
            now = time.time()
            for wind_ind, window_fname in enumerate(windows):
                utils.time_to_go(now, wind_ind, len(windows), 1)
                window = utils.namebase(window_fname)
                powers = epi_utils.concatenate_powers(powers_fol.format(window=window))
                if powersF == 0:
                    powersF = powers.shape[1]
                else:
                    if powers.shape[1] != powersF:
                        print('dims mismatach! {} should be {} for {}'.format(powers.shape[1], powersF, window))
                        continue
                min_f_ind = baselineF - powersF
                norm_powers = (powers - baseline_mean[:, min_f_ind:, :]) / baseline_std[:, min_f_ind:, :]
                norm_powers = norm_powers[:, :, avg_time_crop: -avg_time_crop]
                avg_norm_powers = norm_powers if avg_norm_powers is None else avg_norm_powers + norm_powers
                # avg_norm_powers.append(norm_powers)
            avg_norm_powers /= len(windows)
            # avg_norm_powers = np.array(avg_norm_powers).mean(0)
            print('Saving norm_powers in {}'.format(avg_norm_powers_fname))
            np.save(avg_norm_powers_fname, avg_norm_powers)
        else:
            avg_norm_powers = np.load(avg_norm_powers_fname, mmap_mode='r')
        return avg_norm_powers

    def load_baseline_stat():
        baseline_stat_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power_stat.npz'.format(
            subject, inverse_method, modality, utils.namebase(baseline_window)))
        if not op.isfile(baseline_stat_fname) or overwrite:
            baseline = epi_utils.concatenate_powers(baseline_fol)  # (vertices x freqs x time)
            baseline_std = np.std(baseline, axis=2, keepdims=True)  # the standard deviation (over time) of log baseline values
            baseline_mean = np.mean(baseline, axis=2, keepdims=True)  # the mean (over time) of log baseline values
        else:
            d = np.load(baseline_stat_fname)
            baseline_mean, baseline_std = d['mean'], d['std']
        return baseline_mean, baseline_std

    def find_max_f_t():
        powers_input_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
            subject, inverse_method, modality, '{window}'))
        window_output_fname = powers_input_fname.format(window='{}-avg'.format(power_stc_name))
        if not op.isfile(window_output_fname):
            print('Can\'t find {}!'.format(window_output_fname))
            return

        d = np.load(window_output_fname)
        freqs = epi_utils.get_freqs(d['min_f_ind'] + 1, high_gamma_max)
        times = epi_utils.get_window_times(windows[0], downsample=2)
        times = times[avg_time_crop:-avg_time_crop]
        # t_max = np.where(times > 0.1)[0][0]
        # t_min = 0  # np.where(times > 0)[0][0]
        # x = d['max']  # [:, t_min:t_max]
        # x = np.flip(x, 0)
        # max_val = np.max(x)
        # f_ind, t_ind = np.unravel_index(x.argmax(), x.shape)
        # t_max = t_ind + t_min
        # f_max = f_ind - d['min_f_ind']
        # print('norm_powers_maxs: {:.3f} at {:.2f}s and {}Hz'.format(max_val, t_max, f_max))
        return freqs, times, d['min_f_ind']

    def get_baseline_and_power_file_names():
        files_list_fname = op.join(root_dir, 'labels.pkl')
        if not op.isfile(files_list_fname):
            baseline_files = glob.glob(op.join(baseline_fol, 'epilepsy_*_induced_power.npy'))
            powers_files = glob.glob(op.join(powers_fol, 'epilepsy_*_induced_power.npy'))
            utils.save((baseline_files, powers_files), files_list_fname)
        else:
            baseline_files, powers_files = utils.load(files_list_fname)
        return baseline_files, powers_files

    baseline_files, powers_files = get_baseline_and_power_file_names()
    baseline_mean, baseline_std = load_baseline_stat()
    avg_norm_powers = load_avg_norm_powers(baseline_mean.shape[1])
    freqs, times, min_f_ind = find_max_f_t()
    min_f = min_f_ind + 1

    # Plot the avg
    powers_max = np.max(avg_norm_powers, axis=0)  # over vertices
    powers_min = np.min(avg_norm_powers, axis=0)  # over vertices

    x = np.flip(powers_max, 0)
    max_val = np.max(x)
    f_max, t_max = np.unravel_index(x.argmax(), x.shape)
    print('norm_powers_maxs: {:.3f} at {:.2f}s and {}Hz'.format(max_val, times[t_max], freqs[f_max]))

    psplots.plot_positive_and_negative_power_spectrum(
        powers_min, powers_max, times, '{} {}'.format(modality, 'avg'),
        figure_fname='', high_gamma_max=high_gamma_max, min_f=min_f,
        show_only_sig_in_graph=True)

    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    if len(labels) == 0:
        raise Exception('Can\'t read the {} labels!'.format(atlas))

    vertices, data_indices = defaultdict(list), defaultdict(list)
    start_ind = 0
    for file_ind, (baseline_fname, powers_fname) in enumerate(zip(baseline_files, powers_files)):
        baseline_label_name = utils.namebase(baseline_fname).split('_')[1]
        powers_label_name = utils.namebase(powers_fname).split('_')[1]
        if baseline_label_name != powers_label_name:
            raise Exception('ASDGF@#Q%EGF#Q$T')
        label_name = powers_label_name
        label = [l for l in labels if l.name == label_name][0]
        vertno, src_sel = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        data_indices[label.hemi].extend(range(start_ind, start_ind + len(src_sel)))
        vertices[label.hemi].extend(vertno[0] if label.hemi == 'lh' else vertno[1])
        start_ind += len(src_sel)


    # t_max = np.where(times > 0.01)[0][0]
    # t_min = np.where(times > 0)[0][0]
    # powers_max = powers_max[:, t_min:t_max]
    f_max, t_max = np.unravel_index(powers_max.argmax(), powers_max.shape)
    print('avg_norm_powers max: {:.3f} at {:.2f}s and {}Hz'.format(
        np.max(avg_norm_powers[:, f_max, t_max]), times[t_max], freqs[f_max]))
    vertices_data = {}
    for hemi in utils.HEMIS:
        hemi_data_indices = np.array(data_indices[hemi])
        vertices_data[hemi] = avg_norm_powers[hemi_data_indices, f_max, :]

    combined_stc = meg.creating_stc_obj(
        vertices_data, vertices, subject, tmin=times[0], tstep=times[1] - times[0])
    print('Saving stc file in {}'.format(output_stc_fname))
    combined_stc.save(output_stc_fname)


def calc_labels_connectivity(
        subject, windows, baseline_window, condition, modality, atlas='laus125', func_rois_atlas=True,
        inverse_method='dSPM', low_freq=1, high_freq=120, con_method='wpli2_debiased', con_mode='cwt_morlet',
        n_cycles=7, min_order=1, max_order=100, windows_length=0, windows_shift=0, calc_only_for_all_freqs=False,
        extract_modes=['mean_flip'], overwrite=False, overwrite_connectivity=False, n_jobs=6):
    if len(windows) == 0:
        print('No windows to combine into an epoch object!')
        return

    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    fwd_usingMEG, fwd_usingEEG = meg.get_fwd_flags(modality)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))

    # todo: calc this 10 automatically
    freqs = utils.get_freqs(10, high_freq)
    if calc_only_for_all_freqs:
        bands = {'all': [None, None]}
    else:
        bands = utils.calc_bands(10, high_freq)

    con_indentifer = ''
    if func_rois_atlas:
        template = '{}-epilepsy-{}-{}-{}-*'.format(subject, inverse_method, modality, specific_window)
        con_atlas_files = glob.glob(op.join(SUBJECTS_DIR, subject, 'label', template))
        if len(con_atlas_files) == 0:
            print('Can\'t find func rois atlas! {}'.format(op.join(SUBJECTS_DIR, 'label', template)))
            return False
        atlas = utils.namebase(utils.select_one_file(con_atlas_files))
        con_indentifer = 'func_rois'

    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
    # for connectivity we need shorter names
    labels = epi_utils.shorten_labels_names(labels)
    # Check if can calc the labels info (if not, we want to know now...)
    connectivity.calc_lables_info(subject, atlas, False, [l.name for l in labels], labels)

    # if not overwrite_connectivity:
    #     for cond in ['{}_interictals'.format(condition), '{}_baseline'.format(condition)]:
    #         meg.save_connectivity(
    #             subject, atlas, {cond:1}, modality, extract_modes, con_method, con_indentifer, bands, labels,
    #             reduce_to_3d=True, overwrite=True)
    #     return

    windows_epochs_template = op.join(
        root_dir, '{}-{}-{}-{}-{}-epo.fif'.format(subject, modality, atlas, inverse_method, '{condition}'))
    windows_epochs = epi_utils.combine_windows_into_epochs(windows, windows_epochs_template.format(condition=condition))
    dt = windows_epochs.tmax - windows_epochs.tmin
    baseline_epochs_fname = windows_epochs_template.format(condition=utils.namebase(baseline_window))
    if not op.isfile(baseline_epochs_fname) or overwrite:
        baseline_epoch = epi_utils.combine_windows_into_epochs([baseline_window])
        baseline_epochs = []
        for win_ind in range(len(windows_epochs)):
            tmin = windows_epochs.tmin + win_ind * dt
            tmax = tmin + dt
            if tmax > baseline_epoch.tmax:
                break
            baseline_epoch_crop = baseline_epoch.copy().crop(tmin, tmax)
            demi_epoch = utils.create_epoch(baseline_epoch_crop.get_data(), baseline_epoch_crop.info)
            baseline_epochs.append(demi_epoch.copy())
        baseline_epochs = mne.concatenate_epochs(baseline_epochs, True)
        if baseline_epochs_fname != '':
            print('Saving epochs to {}'.format(baseline_epochs_fname))
            baseline_epochs.save(baseline_epochs_fname)
    else:
        baseline_epochs = mne.read_epochs(baseline_epochs_fname)

    for epochs, cond in zip([windows_epochs, baseline_epochs],
                            ['{}_interictals'.format(condition), '{}_baseline'.format(condition)]):
        meg.calc_labels_connectivity(
            subject, atlas, {cond:1}, subjects_dir=SUBJECTS_DIR, mmvt_dir=MMVT_DIR, inverse_method=inverse_method,
            pick_ori='normal', inv_fname=inv_fname, fwd_usingMEG=fwd_usingMEG, fwd_usingEEG=fwd_usingEEG,
            con_method=con_method, con_mode=con_mode, cwt_n_cycles=n_cycles, overwrite_connectivity=overwrite_connectivity,
            epochs=epochs, bands=bands, cwt_frequencies=freqs, con_indentifer=con_indentifer, labels=labels,
            min_order=min_order, max_order=max_order, downsample=1, windows_length=windows_length,
            windows_shift=windows_shift, n_jobs=n_jobs)


def normalize_connectivity(subject, condition, modality, high_freq=120, con_method='wpli2_debiased',
                           extract_mode='mean_flip', func_rois_atlas=True, divide_by_baseline_std=True,
                           threshold=0, reduce_to_3d=False, overwrite=False, n_jobs=6):
    bands = utils.calc_bands(1, high_freq, include_all_freqs=True)
    if func_rois_atlas:
        con_indentifer = 'func_rois'
    for band_name in bands.keys():
        connectivity_template = connectivity.get_output_fname(
            subject, con_method, modality, extract_mode, '{}_{}_{}'.format(band_name, '{condition}', con_indentifer))
        output_fname = '{}_zvals.npz'.format(
            connectivity_template.format(condition='{}_interictals'.format(condition))[:-4])
        cond_fname = connectivity_template.format(condition='{}_interictals'.format(condition))
        if not op.isfile(cond_fname):
            print('{} is missing!'.format(cond_fname))
            continue

        baseline_fname = connectivity_template.format(condition='{}_baseline'.format(condition))
        if not op.isfile(baseline_fname):
            print('{} is missing!'.format(baseline_fname))
            continue
        print('normalize_connectivity: {} {}:'.format(utils.namebase(cond_fname), band_name))
        d_baseline = utils.Bag(np.load(baseline_fname))
        d_cond = utils.Bag(np.load(cond_fname))
        d_cond.con_values = epi_utils.norm_values(
            d_baseline.con_values, d_cond.con_values, divide_by_baseline_std, threshold, True)
        if 'con_values2' in d_baseline:
            d_cond.con_values2 = epi_utils.norm_values(
                d_baseline.con_values2, d_cond.con_values2, divide_by_baseline_std, threshold, True)
        if reduce_to_3d:
            d_cond.con_values = connectivity.find_best_ord(d_cond.con_values, False)
            d_cond.con_values2 = connectivity.find_best_ord(d_cond.con_values2, False)
        print('Saving norm connectivity in {}'.format(output_fname))
        np.savez(output_fname, **d_cond)


# def norm_values(baseline_x, cond_x, divide_by_baseline_std, threshold, find_best_ord=False):
#     baseline_mean = baseline_x.mean(axis=1, keepdims=True)
#     baseline_std = baseline_x.std(axis=1, keepdims=True) if divide_by_baseline_std else None
#     if threshold > 0:
#         mask_indices = np.where(np.max(np.abs(cond_x), axis=1) < threshold)
#     if divide_by_baseline_std:
#         cond_x = (cond_x - baseline_mean) / baseline_std
#     else:
#         cond_x = cond_x - baseline_mean
#     if threshold > 0:
#         cond_x[mask_indices[0], :, mask_indices[1]] = np.zeros(cond_x.shape[1])
#     if find_best_ord:
#         new_con_x = np.zeros((cond_x.shape[0], cond_x.shape[1]))
#         for n in range(cond_x.shape[0]):
#             best_ord = np.argmax(np.abs(cond_x[n]).max(0))
#             new_con_x[n] = cond_x[n, :, best_ord]
#     print('{:.4f} {:.4f}'.format(np.nanmin(cond_x), np.nanmax(cond_x)))
#     return cond_x


def find_functional_rois(subject, condition, modality, atlas='laus125', min_cluster_size=10, inverse_method='dSPM'):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    fwd_usingMEG, fwd_usingEEG = meg.get_fwd_flags(modality)
    stcs_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    stc_template = op.join(stcs_fol, '{}-epilepsy-{}-{}-{}-*zvals*-rh.stc'.format(subject, inverse_method, modality, condition))
    stc_name = utils.select_one_file(glob.glob(stc_template), stc_template)[:-len('-rh.stc')]
    if not utils.stc_exist(stc_name):
        return
    meg.find_functional_rois_in_stc(
        subject, subject, atlas, stc_name, 0, threshold_is_precentile=False, extract_time_series_for_clusters=False,
        min_cluster_size=min_cluster_size, inv_fname=inv_fname, fwd_usingMEG=fwd_usingMEG, fwd_usingEEG=fwd_usingEEG,
        time_index=0, abs_max=False, modality=modality, n_jobs=n_jobs)


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_template,
         inverse_method='dSPM', specific_window='', exclude_windows=[], no_runs=False, recursive=False,
         check_windows=True, atlas='aparc.DKTatlas40', n_jobs=4):
    # run_num = re.sub('\D', ',', run).split(',')[-1].zfill(2)
    if no_runs:
        if recursive:
            windows = glob.glob(op.join(evokes_fol, '**', '*.fif'), recursive=True)
        else:
            windows = glob.glob(op.join(evokes_fol, '*.fif'))
        baseline_windows = glob.glob(op.join(evokes_fol, '*{}*.fif'.format(baseline_template)))
    else:
        windows = glob.glob(op.join(evokes_fol, '{}_*.fif'.format(run)))
        baseline_windows = glob.glob(op.join(evokes_fol, '{}_{}*.fif'.format(run, baseline_template)))
    for baseline_window in baseline_windows:
        if baseline_window in windows:
            windows.remove(baseline_window)
    if specific_window != '':
        windows = [w for w in windows if specific_window in utils.namebase(w)]
    if len(exclude_windows) > 0:
        windows = [w for w in windows if all([
            exclude_window not in utils.namebase(w) for exclude_window in exclude_windows])]
    if len(windows) == 0:
        print('No windows!')
    baseline_window = utils.select_one_file(baseline_windows, 'baseline')
    baseline_windows = [baseline_window]
    windows_with_baseline = windows + baseline_windows
    if len(windows) == 0:
        windows = baseline_windows

    if check_windows:
        print('windows:')
        for ind, window_fname in enumerate(windows):
            print('{}) {}'.format(ind, window_fname))
        print('Baseline: {}'.format(baseline_window))
        ret = input('Continue? (y/n) ')
        if not au.is_true(ret):
            return

    baseline_name = utils.namebase(baseline_window)
    overwrite_inv = False
    overwrite_fwd = False
    overwrite_evokes = True
    overwrite_plots = False
    check_for_labels_files = True
    overwrite_induced_power_zvals = False
    overwrite_stc = False
    overwrite_modalities_figures = False
    from_index, to_index = None, None # 2000, 10000
    max_t = 0 #7500
    high_freq = 120
    low_freq = 1
    percentiles = [5, 95]
    sig_threshold = 2
    figures_type = 'jpg'
    save_fig = True
    plot_baseline_stat = False
    avg_use_abs = False
    avg_time_crop = 100
    power_specturm_win_suffix = '-avg'
    con_method =  'gc' # 'granger-causality' # 'wpli2_debiased'
    con_mode = 'cwt_morlet'
    con_atlas = 'laus125'
    min_cluster_size = 10
    bad_channels = bad_channels.split(',')

    # epi_utils.create_evokeds_links(subject, windows_with_baseline)
    for modality in modalities:
        # 1) Sensors
        # plots.plot_sensors_windows(subject, windows, specific_window, modality, bad_channels)
        # plots.plot_average_sensors(subject, windows, specific_window, modality, bad_channels)
        # plots.plot_evokes(subject, modality, windows, bad_channels, n_jobs > 1, overwrite_evokes)
        # plots.plot_topomaps(subject, modality, windows, bad_channels, parallel=n_jobs > 1)
        # calc_sensors_power(subject, windows_with_baseline, modality, inverse_method, bad_channels,
        #                    high_gamma_max=high_freq, downsample=2, parallel=n_jobs > 1, overwrite=False)
        # psplots.plot_sensors_powers(
        #     subject, windows, baseline_window, modality, inverse_method, high_gamma_max=high_freq,
        #     percentiles=percentiles, sig_threshold=sig_threshold, save_fig=save_fig,
        #     plot_baseline_stat=plot_baseline_stat, bad_channels=bad_channels, overwrite=False, parallel=False)

        # 2) calc fwd and inv
        # calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels,
        #              overwrite_inv=overwrite_inv, overwrite_fwd=overwrite_fwd)
        # check_inv_fwd(subject, modality, run_num)

        # 3) Amplitude
        calc_amplitude(subject, modality, run_num, windows_with_baseline, inverse_method, overwrite_stc, True, n_jobs)
        calc_amplitude_zvals(
            subject, windows, baseline_name, modality, from_index, to_index, inverse_method,
            use_abs=False, parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)
        average_amplitude_zvals(subject, windows, modality, specific_window, avg_use_abs, inverse_method='dSPM',
                                do_plot=True, overwrite=True)
        find_functional_rois(subject, specific_window, modality, con_atlas, min_cluster_size, inverse_method)
        calc_labels_connectivity(
            subject, windows, baseline_window, specific_window, modality, con_atlas, True, inverse_method,
            low_freq, high_freq, con_method, con_mode, n_cycles=2, min_order=1, max_order=20,
            windows_length=100, windows_shift=10, calc_only_for_all_freqs=True, overwrite=True,
            overwrite_connectivity=False, n_jobs=n_jobs)
        normalize_connectivity(
            subject, specific_window, modality, high_freq, con_method, divide_by_baseline_std=False,
            threshold=0.5, reduce_to_3d=True, overwrite=True, n_jobs=n_jobs)
        plots.plot_connectivity(subject, specific_window, modality, high_freq, con_method)

        # 4) Induced power
        # calc_induced_power(subject, run_num, windows_with_baseline, modality, inverse_method, check_for_labels_files,
        #                    overwrite=True)
        # psplots.plot_powers(subject, windows, modality, inverse_method, high_freq, figures_type,
        #         overwrite=False)
        # psplots.plot_baseline_source_powers(
        #     subject, baseline_window, modality, inverse_method, high_freq, figures_type, overwrite_plots)
        # psplots.plot_norm_powers(
        #     subject, windows, baseline_window, modality, inverse_method, figures_type=figures_type, overwrite=False)
        # psplots.average_norm_powers(
        #     subject, windows, modality, specific_window, inverse_method, avg_time_crop, overwrite=True,
        #     save_fig=True, figures_type=figures_type)
        # psplots.plot_norm_powers_per_label(subject, windows, baseline_window, modality, inverse_method,
        #                            calc_also_non_norm_powers=False, overwrite=True, n_jobs=n_jobs)
        # calc_stc_power_specturm(
        #     subject, modality, specific_window, windows[0], baseline_window, avg_time_crop, run_num)
        # calc_avg_power_specturm_stc(
        #     subject, modality, specific_window, windows, baseline_window, avg_time_crop, run_num,
        #     inverse_method, atlas, high_freq)

        # 5) Connectivity
        # calc_labels_connectivity(
        #     subject, windows, baseline_window, specific_window, modality, con_atlas, False, inverse_method,
        #     low_freq, high_freq, con_method, con_mode, n_cycles=2,
        #     overwrite=False, n_jobs=n_jobs)
        # normalize_connectivity(subject, specific_window, modality, high_freq, con_method,
        #                        overwrite=False, n_jobs=n_jobs)
        pass

    # find_vertices(subject, run_num)
    # for window_fname in windows:
    #     figure_name = '' #'{}-modalities-power-spectrum-with-grpahs'.format(utils.namebase(window_fname))
    #     psplots.plot_modalities_power_spectrums_with_graph(subject, modalities, window_fname, figure_name, file_type='eps')

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


def all_conditions_main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_template,
         inverse_method='dSPM', specific_windows=[], exclude_windows=[], no_runs=False, recursive=False,
         check_windows=True, atlas='aparc.DKTatlas40', n_jobs=4):
    high_freq = 120
    con_method = 'gc'  # 'granger-causality' # 'wpli2_debiased'
    extract_mode = 'mean_flip'
    for modality in modalities:
        plots.plot_both_conditions(
            subject, specific_windows, modality, high_freq, con_method, extract_mode, func_rois_atlas=True,
            node_name='occipital', use_zvals=False, windows_len=100, windows_shift=10)
        # pass


if __name__ == '__main__':
    import argparse
    from src.examples.epilepsy import init_files
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-n', '--n_jobs', help='cpu num', required=True)
    args = utils.Bag(au.parse_parser(parser))
    n_jobs = utils.get_n_jobs(args.n_jobs)

    modalities = ['meg', 'eeg', 'meeg']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    inverse_method = 'dSPM'
    atlas = 'aparc.DKTatlas40'
    recursive = False
    check_windows = False
    subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name, no_runs = init_files.subject_nmr01327()
    run_files = [utils.namebase(f).split('_')[0] for f in glob.glob(op.join(evokes_fol, 'run*_*.fif'))]
    if recursive:
        evokes_files = glob.glob(op.join(evokes_fol, '**', '*.fif'), recursive=True)
    else:
        evokes_files = glob.glob(op.join(evokes_fol, '*.fif'))
    runs = []
    if len(run_files) == len(evokes_files):
        runs = set(run_files)
    if len(runs) == 0 or no_runs:
        print('No run were found!')
        runs = ['01']
    print('n_jobs: {}'.format(n_jobs))
    specific_windows = ['R', 'L'] # 'L', # ['baseline_run1_195'] # ['L', 'R'] # 'MEG_SZ_run1_107.7_11sec' # 'sz_1.3s' # '550_20sec'#  #'bl_474s' #  #' # 'sz_1.3s' #'550_20sec' #  'bl_474s' # 'run2_bl_248s'
    exclude_windows = []#['baseline_run1_SHORT_600ms', 'MEG_SZ_run1_108.6', 'MEG_SZ_run1_107.7_11se',
                       # 'EEG_SZ_run1_114.3_11sec', 'EEG_SZ_run1_114.3']
    for run in runs:
        # if run != 'run1':
        #     continue
        raw_fname, run_num = init_files.find_raw_fname(meg_fol, run)
        # if len(runs) > 0:
        if not op.isfile(raw_fname):
            ret = input('No raw file! Do you want to continue (y/n)? ')
            if not au.is_true(ret):
                continue
        for specific_window in specific_windows:
            main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_name,
                 inverse_method, specific_window, exclude_windows, no_runs, recursive, check_windows, atlas, n_jobs)
        # all_conditions_main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_name,
        #          inverse_method, specific_windows, exclude_windows, no_runs, recursive, check_windows, atlas, n_jobs)
    print('Finish!')