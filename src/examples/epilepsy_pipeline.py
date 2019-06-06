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
import re

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


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
    print('Calculating max power for {}'.format(fol))
    max_powers = np.max(np.concatenate([np.load(powers_fname) for powers_fname in powers_files]), axis=0)
    print('Saving to {}'.format(output_fname))
    np.save(output_fname, max_powers)


def plot_max_powers(subject, windows_fnames, modality, inverse_method='dSPM', overwrite=False, parallel=True):
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, 120, 5)])
    params = [(subject, window_fname, modality, inverse_method, freqs, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_plot_max_powers_parllel, params, len(windows_fnames) if parallel else 1)


def _plot_max_powers_parllel(p):
    subject, window_fname, modality, inverse_method, freqs, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_max_power.npy'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if not op.isfile(fname):
        print('Can\'t find {}!'.format(fname))
        return
    # /homes/5/npeled/space1/mmvt/nmr01321/epilepsy-figures/power-spectrum
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    figure_fname = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-induced_max_power.jpg'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    # if op.isfile(figure_fname) and not overwrite:
    #     return
    powers = np.load(fname)
    times = np.arange(1, powers.shape[1])
    plot_power_spectrum(powers.astype(np.float32).T, times, freqs, figure_fname)


def plot_power_spectrum(powers, times, freqs, figure_fname):
    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    plt.figure()
    plt.imshow(np.flip(powers.T, 0), aspect='auto', interpolation='nearest', extent=extents(times) + extents(freqs), cmap='hot')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (seconds)')
    plt.tight_layout()
    plt.savefig(figure_fname)
    plt.figure()
    plt.plot(powers)
    print('asdf')


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


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_template,
         inverse_method='dSPM', n_jobs=4):
    run_num = re.sub('\D', ',', run).split(',')[-1]
    windows = glob.glob(op.join(evokes_fol, '{}_*.fif'.format(run)))
    baseline_windows = glob.glob(op.join(evokes_fol, '{}_{}*.fif'.format(run, baseline_template)))
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

    # create_evokeds_links(subject, windows_with_baseline)
    for modality in modalities:
        # calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels,
        #              overwrite_inv=overwrite_inv, overwrite_fwd=overwrite_fwd)
        # check_inv_fwd(subject, modality, run_num)
        # plot_evokes(subject, modality, windows, bad_channels, n_jobs > 1, overwrite_evokes)
        # plot_topomaps(subject, modality, windows, bad_channels, parallel=n_jobs > 1)
        # calc_amplitude(subject, modality, run_num, windows_with_baseline, inverse_method, overwrite_stc, n_jobs)
        calc_induced_power(subject, run_num, windows_with_baseline, modality, inverse_method, check_for_labels_files,
                           overwrite_stc)
        # calc_max_powers(subject, windows_with_baseline, modality, inverse_method, overwrite=False, parallel=True)
        # plot_max_powers(subject, windows_with_baseline, modality, inverse_method, overwrite=False, parallel=False)
        # calc_amplitude_zvals(
        #     subject, windows, baseline_name, modality, from_index, to_index, inverse_method,
        #     parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)
        # calc_induced_power_zvals(
        #     subject, windows, baseline_name, modality, bands, from_index, to_index, inverse_method,
        #     parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)
        # move_non_zvals_stcs(subject, modality)

        # plot_stcs_files(subject, modality, n_jobs)
        # plot_windows(subject, windows, modality, bands, inverse_method)
        # plot_freqs(subject, temporal_windows, modality, bands, inverse_method, max_t)
        pass

    # plot_modalities(subject, windows, modalities, bands, inverse_method, max_t, overwrite_modalities_figures, n_jobs)
    # plot_activity_modalities(subject, windows, modalities, inverse_method, overwrite=overwrite_modalities_figures)
    # plot_baseline(subject, baseline_name)
    # fix_amplitude_fnames(subject, bands)



if __name__ == '__main__':
    # subject = 'nmr00857'
    # windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    # windows += ['/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/43.9s.fif']
    # baseline_windows = ['/autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/37.3_BGprSzs.fif']
    # temporal_windows = [w for w in windows if '_Ts' in utils.namebase(w)]
    # frontal_windows = [w for w in windows if '_Fs' in utils.namebase(w)]
    # baseline_name = '37.3_BGprSzs'
    # bad_channels = 'EEG061,EEG02,EEG042,MEG0112,MEG0113'

    modalities = ['eeg', 'meg', 'meeg']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

    subject = 'nmr01321'
    inverse_method = 'dSPM'
    evokes_fol = [d for d in [
        # '/autofs/space/frieda_001/users/valia/epilepsy/4272326_01321/MMVT_epochs',
        # '/homes/5/npeled/space1/MEG/nmr01321/evokeds',
        op.join(MMVT_DIR, subject, 'evoked')] if op.isdir(d)][0]
    meg_fol = [d for d in [
        '/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123',
        op.join(MEG_DIR, subject)] if op.isdir(d)][0]
    empty_fname = glob.glob(op.join(meg_fol, '*roomnoise_raw.fif'))[0]
    bad_channels = 'EEG001,EEG003,EEG004,EEG005,EEG008,EEG034,EEG045,EEG051,EEG057,EEG058,EEG060,EEG061,EEG062,EEG074,MEG1422,MEG1532,MEG2012,MEG2022'

    runs = set([utils.namebase(f).split('_')[0] for f in glob.glob(op.join(evokes_fol, 'run*_*.fif'))])
    if len(runs) == 0:
        print('No run were found!')
        runs = ['no_runs']
    n_jobs = 5 # utils.get_n_jobs(-5)
    print('n_jobs: {}'.format(n_jobs))
    for run in runs:
        run_num = re.sub('\D', ',', run).split(',')[-1]
        if int(run_num) != 1:
            continue
        raw_fname = glob.glob(op.join(meg_fol, '*_{}_raw.fif'.format(str(run_num).zfill(2))))[0]
        main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, 'Base_line',
             inverse_method, n_jobs)
    print('Finish!')