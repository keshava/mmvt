import os.path as op
from src.utils import utils
from src.utils import labels_utils as lu
from src.examples.epilepsy import pipeline
from src.examples.epilepsy import utils as epi_utils
from src.preproc import meg, eeg, connectivity
import glob
import mne
import numpy as np

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
ELECTRODES_DIR = utils.get_link_dir(LINKS_DIR, 'electrodes')


def find_remote_subject_dir(subject):
    remote_subjects_dir = '/space/megraid/clinical/MEG-MRI/seder/freesurfer'
    if not op.isdir(op.join(remote_subjects_dir, subject)):
        remote_subjects_dir = SUBJECTS_DIR
    remote_subject_dir = op.join(remote_subjects_dir, subject)
    if not op.isdir(remote_subject_dir):
        raise Exception('No reocon-all files!')
    # print(('No reocon-all files!'))
    return remote_subject_dir


def init_nmr01391():
    subject = 'nmr01391'
    remote_subject_dir = find_remote_subject_dir(subject)
    meg_fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/1391'
    if not op.isdir(meg_fol):
        meg_fol = op.join(MEG_DIR, subject)
    bad_channels = ['MEG{}'.format(c) for c in [
        '0113', '1532', '1623', '2042', '1912', '2032', '2522', '0642', '0121', '1421', '1221', '1023',
        '0741', '1022', '1242']]
    raw_fname = op.join(meg_fol, 'raw', '6859241_03_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, 'raw', '6859241_emptyroom_raw.fif')
    return subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


# def average_baseline(subject, inverse_method, modality, overwrite=True):
#     fol = op.join(MMVT_DIR, subject, modality, 'baseline-{}-stcs'.format(inverse_method))
#     baseline_stcs = glob.glob(op.join(fol, '{}-epilepsy-{}-{}-*-rh.stc'.format(subject, inverse_method, modality)))
#     output_fname = op.join(fol, '{}-baseline-{}-{}'.format(subject, inverse_method, modality))
#     if stc_exist(output_fname) and not overwrite:
#         print('baseline average already exist')
#         return output_fname
#     stcs = [mne.read_source_estimate(stc_fname) for stc_fname in baseline_stcs]
#     data = np.array([stc.data for stc in stcs]).mean(axis=0)
#     baseline_mean = mne.SourceEstimate(data, stcs[0].vertices, stcs[0].tmin, stcs[0].tstep, subject=subject)
#     baseline_mean.save(output_fname)
#     return output_fname


def stc_exist(stc_name):
    return utils.both_hemi_files_exist('{}-{}.stc'.format(stc_name, '{hemi}'))


def calc_stcs(subject, modality, clips_dict, inverse_method='MNE', downsample_r=2, overwrite=False, n_jobs=4):
    for clip_type, clips in clips_dict.items():
        new_fol = utils.make_dir(op.join(
            MMVT_DIR, subject, modality, '{}-{}-stcs'.format(clip_type, inverse_method)))
        pipeline.calc_amplitude(
            subject, modality, clips, inverse_method, downsample_r, overwrite, False, n_jobs)
        for clip_fname in clips:
            template = '{}-epilepsy-{}-{}-{}-?h.stc'.format(
                subject, inverse_method, modality, utils.namebase(clip_fname))
            stc_fnames = glob.glob(op.join(MMVT_DIR, subject, modality, template))
            if len(stc_fnames) != 2:
                print('**** Error with {} ({} files) ****'.format(template, len(stc_fnames)))
            utils.move_files(stc_fnames, new_fol, overwrite)


def calc_zvals(subject, modality, ictal_clips, inverse_method, overwrite=False, n_jobs=4):
    from_index, to_index = None, None
    use_abs = False
    baseline_stc_fnames = glob.glob(op.join(
        MMVT_DIR, subject, meg.modality_fol(modality), 'baseline-{}-stcs'.format(inverse_method),
        '{}-epilepsy-{}-{}-*-rh.stc'.format(subject, inverse_method, modality)))
    if len(baseline_stc_fnames) == 0:
        print('No baseline!')
        return False
    fol = utils.make_dir(op.join(
        MMVT_DIR, subject, meg.modality_fol(modality), 'ictal-{}-zvals-stcs'.format(inverse_method)))
    params = [(subject, clip_fname, baseline_stc_fnames, inverse_method, fol, from_index, to_index, use_abs, overwrite)
              for clip_fname in ictal_clips]
    utils.run_parallel(_calc_zvals_parallel, params, n_jobs)


def _calc_zvals_parallel(p):
    subject, clip_fname, baseline_fnames, inverse_method, fol, from_index, to_index, use_abs, overwrite = p
    stc_zvals_fname = op.join(fol, '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
        subject, inverse_method, modality, utils.namebase(clip_fname)))
    if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_zvals_fname, '{hemi}')) and not overwrite:
        return True
    stc_fname = op.join(
        MMVT_DIR, subject, meg.modality_fol(modality), 'ictal-{}-stcs'.format(inverse_method),
        '{}-epilepsy-{}-{}-{}'.format(subject, inverse_method, modality, utils.namebase(clip_fname)))
    if not utils.both_hemi_files_exist('{}-{}.stc'.format(stc_fname, '{hemi}')):
        print('Error finding {}!'.format(stc_fname))
        return False
    return meg.calc_stc_zvals(
        subject, '{}-rh.stc'.format(stc_fname), baseline_fnames, stc_zvals_fname, # '{}-rh.stc'.format(baseline_fname)
        use_abs, from_index, to_index, True, overwrite)


def find_functional_rois(subject, ictal_clips, modality, seizure_times=(0, 0.1), atlas='laus125',
                         min_cluster_size=10, inverse_method='MNE', overwrite=False, n_jobs=4):
    fwd_usingMEG, fwd_usingEEG = meg.get_fwd_flags(modality)
    stcs_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality), 'ictal-{}-zvals-stcs'.format(inverse_method))
    # Make sure we have a morph map, and if not, create it here, and not in the parallel function
    mne.surface.read_morph_map(subject, subject, subjects_dir=SUBJECTS_DIR)
    for clip_fname in ictal_clips:
        stc_name = op.join(stcs_fol, '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
            subject, inverse_method, modality, utils.namebase(clip_fname)))
        if not stc_exist(stc_name):
            return False
        stc = mne.read_source_estimate('{}-rh.stc'.format(stc_name))
        stc.crop(stc.tmin, 0)
        mean_baseline = np.max(stc.data, axis=0).squeeze().mean()
        meg.find_functional_rois_in_stc(
            subject, subject, atlas, stc_name, mean_baseline, threshold_is_precentile=False, extract_time_series_for_clusters=False,
            min_cluster_size=min_cluster_size, min_cluster_max=mean_baseline, fwd_usingMEG=fwd_usingMEG,
            fwd_usingEEG=fwd_usingEEG, time_index=0, abs_max=False, modality=modality,
            crop_times=seizure_times, avg_stc=True, n_jobs=n_jobs)


def calc_rois_connectivity(subject, ictal_clips, modality, inverse_method, min_order=1, max_order=20,
                           windows_length=100, windows_shift=10, overwrite=False, n_jobs=4):
    fwd_usingMEG, fwd_usingEEG = meg.get_fwd_flags(modality)
    bands = {'all': [None, None]}
    clusters_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality), 'clusters')
    for clip_fname in ictal_clips:
        labels_fol = op.join(
            clusters_fol, '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
                subject, inverse_method, modality, utils.namebase(clip_fname)))
        labels = lu.read_labels_files(subject, labels_fol, n_jobs=n_jobs)
        # for connectivity we need shorter names
        labels = epi_utils.shorten_labels_names(labels)
        # Check if can calc the labels info (if not, we want to know now...)
        connectivity.calc_lables_info(subject, labels=labels)
        evoked = mne.read_evokeds(clip_fname)[0]
        cond = atlas = utils.namebase(clip_fname)
        meg.calc_labels_connectivity(
            subject, atlas, {cond: 1}, subjects_dir=SUBJECTS_DIR, mmvt_dir=MMVT_DIR, inverse_method=inverse_method,
            pick_ori='normal', fwd_usingMEG=fwd_usingMEG, fwd_usingEEG=fwd_usingEEG,
            con_method='gc', overwrite_connectivity=overwrite,
            epochs=evoked, bands=bands, con_indentifer='func_rois', labels=labels,
            min_order=min_order, max_order=max_order, downsample=1, windows_length=windows_length,
            windows_shift=windows_shift, n_jobs=n_jobs)

    # windows_epochs_template = op.join(
    #     root_dir, '{}-{}-{}-{}-{}-epo.fif'.format(subject, modality, atlas, inverse_method, '{condition}'))
    # windows_epochs = epi_utils.combine_windows_into_epochs(windows, windows_epochs_template.format(condition=condition))
    # dt = windows_epochs.tmax - windows_epochs.tmin
    # baseline_epochs_fname = windows_epochs_template.format(condition=utils.namebase(baseline_window))
    # if not op.isfile(baseline_epochs_fname) or overwrite:
    #     baseline_epoch = epi_utils.combine_windows_into_epochs([baseline_window])
    #     baseline_epochs = []
    #     for win_ind in range(len(windows_epochs)):
    #         tmin = windows_epochs.tmin + win_ind * dt
    #         tmax = tmin + dt
    #         if tmax > baseline_epoch.tmax:
    #             break
    #         baseline_epoch_crop = baseline_epoch.copy().crop(tmin, tmax)
    #         demi_epoch = utils.create_epoch(baseline_epoch_crop.get_data(), baseline_epoch_crop.info)
    #         baseline_epochs.append(demi_epoch.copy())
    #     baseline_epochs = mne.concatenate_epochs(baseline_epochs, True)
    #     if baseline_epochs_fname != '':
    #         print('Saving epochs to {}'.format(baseline_epochs_fname))
    #         baseline_epochs.save(baseline_epochs_fname)
    # else:
    #     baseline_epochs = mne.read_epochs(baseline_epochs_fname)
    #
    # for epochs, cond in zip([windows_epochs, baseline_epochs],
    #                         ['{}_interictals'.format(condition), '{}_baseline'.format(condition)]):
    #     meg.calc_labels_connectivity(
    #         subject, atlas, {cond:1}, subjects_dir=SUBJECTS_DIR, mmvt_dir=MMVT_DIR, inverse_method=inverse_method,
    #         pick_ori='normal', inv_fname=inv_fname, fwd_usingMEG=fwd_usingMEG, fwd_usingEEG=fwd_usingEEG,
    #         con_method=con_method, con_mode=con_mode, cwt_n_cycles=n_cycles, overwrite_connectivity=overwrite_connectivity,
    #         epochs=epochs, bands=bands, cwt_frequencies=freqs, con_indentifer=con_indentifer, labels=labels,
    #         min_order=min_order, max_order=max_order, downsample=1, windows_length=windows_length,
    #         windows_shift=windows_shift, n_jobs=n_jobs)


def main(subject, modality, clips_dict, inverse_method='MNE', downsample_r=2, seizure_times=(0, .1), atlas='laus125',
         min_cluster_size=10, overwrite=False, min_order=1, max_order=20,
        windows_length=100, windows_shift=10, n_jobs=4):
    calc_stcs(subject, modality, clips_dict, inverse_method, downsample_r, overwrite=True, n_jobs=n_jobs)
    # calc_zvals(subject, modality, clips_dict['ictal'], inverse_method, overwrite=True, n_jobs=n_jobs)
    # find_functional_rois(
    #     subject, clips_dict['ictal'], modality, seizure_times, atlas, min_cluster_size,
    #     inverse_method, overwrite=True, n_jobs=n_jobs)
    # calc_rois_connectivity(subject, clips_dict['ictal'], modality, inverse_method, min_order, max_order,
    #                        windows_length, windows_shift, overwrite, n_jobs)


if __name__ == '__main__':
    n_jobs = utils.get_n_jobs(-10)
    n_jobs = n_jobs if n_jobs > 1 else 1
    print('{} jobs'.format(n_jobs))
    modality = 'meg'
    fif_files, clips_dict = [], {}

    subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01391()
    for subfol in ['baseline', 'ictal']:
        files = glob.glob(op.join(meg_fol, subfol, '*.fif'))
        fif_files += files
        clips_dict[subfol] = files

    main(subject, modality, clips_dict, inverse_method='MNE', downsample_r=2, seizure_times=(0, .1),
         atlas='laus125', min_cluster_size=10, overwrite=False, n_jobs=n_jobs)