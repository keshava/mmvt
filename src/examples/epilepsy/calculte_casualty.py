import os
import os.path as op
from src.utils import utils
from src.utils import labels_utils as lu
from src.examples.epilepsy import pipeline
from src.examples.epilepsy import utils as epi_utils
from src.examples.epilepsy import plots
from src.preproc import meg, eeg, connectivity
from src.preproc import anatomy as anat
import glob
import traceback
import mne
import numpy as np
from tqdm import tqdm


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
    mri_subject, meg_subject = 'nmr01391', 'nmr01391'
    remote_subject_dir = find_remote_subject_dir(mri_subject)
    meg_fol = '/autofs/space/frieda_001/users/valia/mmvt_root/meg/1391'
    if not op.isdir(meg_fol):
        meg_fol = op.join(MEG_DIR, meg_subject)
    bad_channels = ['MEG{}'.format(c) for c in [
        '0113', '1532', '1623', '2042', '1912', '2032', '2522', '0642', '0121', '1421', '1221', '1023',
        '0741', '1022', '1242']]
    raw_fname = op.join(meg_fol, 'raw', '6859241_03_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, 'raw', '6859241_emptyroom_raw.fif')
    return mri_subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


def init_nmr01417():
    mri_subject, meg_subject = 'nmr01417', 'nmr01417'
    remote_subject_dir = find_remote_subject_dir(mri_subject)
    meg_fol = '/autofs/space/frieda_003/users/valia/epilepsy_clin/5904438_01417/200611'
    if not op.isdir(meg_fol):
        meg_fol = op.join(MEG_DIR, meg_subject)
    bad_channels = []
    raw_fname = op.join(meg_fol, '5904438_01_raw_ssst.fif')
    empty_room_fname = op.join(meg_fol, 'raw', '6859241_emptyroom_raw.fif')
    return mri_subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname


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


def calc_baseline_threshold(subject, modality, inverse_method, overwrite=False):
    baseline_threshold_fname = op.join(
        MMVT_DIR, subject, meg.modality_fol(modality), 'baseline-{}-stcs'.format(inverse_method),
        'averaged_threhold.npy')
    if op.isfile(baseline_threshold_fname) and not overwrite:
        return np.load(baseline_threshold_fname)

    baseline_stc_fnames = glob.glob(op.join(
        MMVT_DIR, subject, meg.modality_fol(modality), 'baseline-{}-stcs'.format(inverse_method),
        '{}-epilepsy-{}-{}-*-rh.stc'.format(subject, inverse_method, modality)))
    baseline_data = np.array([mne.read_source_estimate(baseline_fname).data for baseline_fname in baseline_stc_fnames])
    N, V, T = baseline_data.shape
    baseline_data = np.reshape(baseline_data, (V, N * T))
    baseline_threshold = np.median(np.max(baseline_data, axis=0).squeeze())
    print('baseline threshold: {}'.format(baseline_threshold))
    np.save(baseline_threshold_fname, baseline_threshold)
    return baseline_threshold


def calc_stc_zvals(subject, modality, ictal_clips, inverse_method, overwrite=False, n_jobs=4):
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
    params = [(subject, clip_fname, baseline_stc_fnames, modality, inverse_method, fol, from_index, to_index, use_abs,
               overwrite) for clip_fname in ictal_clips]
    utils.run_parallel(_calc_zvals_parallel, params, n_jobs)


def _calc_zvals_parallel(p):
    subject, clip_fname, baseline_fnames, modality, inverse_method, fol, from_index, to_index, use_abs, overwrite = p
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


def calc_accumulate_stc_as_time(subject, ictal_clips, modality, seizure_times, windows_length, windows_shift,
                                mean_baseline, inverse_method, n_jobs):
    modality_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality))
    stcs_fol = utils.make_dir(op.join(modality_fol, 'time_accumulate'))
    windows = calc_windows(seizure_times, windows_length, windows_shift)
    for ictal_clip in ictal_clips:
        output_fname = op.join(stcs_fol, '{}_labels_times.txt'.format(utils.namebase(ictal_clip)))
        output_str = '{}:\n'.format(utils.namebase(ictal_clip))
        for from_t, to_t in windows:
            ictals = calc_accumulate_stc(
                subject, [ictal_clip], modality, (from_t, to_t), mean_baseline, inverse_method, False, True, n_jobs)
            ictal_fname, stc_name, ictal_stc, mean_baseline, labels_times = ictals[0]
            utils.save(labels_times, op.join(stcs_fol, '{}_{:.2f}_{:.2f}_labels_times.pkl'.format(
                utils.namebase(stc_name), from_t, to_t)))
            for label_name, label_time in labels_times.items():
                output_str += '{}: {:.4f}\n'.format('_'.join(label_name.split('_')[-2:]), label_time)
        # stc_output_fname = op.join(stcs_fol, '{}-time-acc'.format(utils.namebase(stc_name)))
        # print('Saving accumulate stc: {}'.format(stc_output_fname))
        # ictal_stc.save(stc_output_fname)
        print('Saving {}'.format(output_fname))
        with open(output_fname, 'w') as output_file:
            print(output_str, file=output_file)


def calc_accumulate_stc(
        subject, ictal_clips, modality, seizure_times, mean_baseline, inverse_method, reverse, set_t_as_val, n_jobs):
    modality_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality))
    stcs_fol = op.join(modality_fol, 'ictal-{}-zvals-stcs'.format(inverse_method))
    params = [(subject, clip_fname, inverse_method, modality, seizure_times, mean_baseline, stcs_fol, reverse,
               set_t_as_val, n_jobs) for clip_fname in ictal_clips]
    ictals = utils.run_parallel(_calc_ictal_and_baseline_parallel, params, n_jobs)
    return ictals


def calc_windows(seizure_times, windows_length, windows_shift):
    return utils.create_windows(
        seizure_times[1] - seizure_times[0] + windows_length, windows_length, windows_shift) + seizure_times[0]


def find_functional_rois(
        subject, ictal_clips, modality, seizure_times, atlas, min_cluster_size, inverse_method, mean_baseline,
        windows_length=0.1, windows_shift=0.05, overwrite=False, n_jobs=4):
    modality_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality))
    # Make sure we have a morph map, and if not, create it here, and not in the parallel function
    mne.surface.read_morph_map(subject, subject, subjects_dir=SUBJECTS_DIR)
    connectivity = anat.load_connectivity(subject)
    if overwrite:
        utils.delete_folder_files(op.join(MMVT_DIR, subject, modality_fol, 'clusters'))
    windows = calc_windows(seizure_times, windows_length, windows_shift)
    params = []
    for ictal_clip in ictal_clips:
        for from_t, to_t in windows:
            params.append((subject, modality, ictal_clip, atlas, inverse_method, mean_baseline, from_t, to_t,
                           min_cluster_size, connectivity))
    utils.run_parallel(_find_functional_rois_parallel, params, n_jobs)


def _find_functional_rois_parallel(p):
    (subject, modality, ictal_clip, atlas, inverse_method, mean_baseline, from_t, to_t, min_cluster_size,\
     connectivity) = p
    fwd_usingMEG, fwd_usingEEG = meg.get_fwd_flags(modality)
    # modality_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality))
    # ictlas_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-amplitude-zvals-ictals.pkl'.format(
    #     subject, inverse_method, modality))

    ictals = calc_accumulate_stc(
        subject, [ictal_clip], modality, (from_t, to_t), mean_baseline, inverse_method, True, False, 1)
    _, stc_name, ictal_stc, mean_baseline, _ = ictals[0]
    max_ictal = ictal_stc.data.max()
    if max_ictal < mean_baseline:
        print('max ictal ({}) < mean baseline ({})!'.format(max_ictal, mean_baseline))
        return False
    meg.find_functional_rois_in_stc(
        subject, subject, atlas, utils.namebase(stc_name), mean_baseline, threshold_is_precentile=False,
        extract_time_series_for_clusters=False, time_index=0, min_cluster_size=min_cluster_size,
        min_cluster_max=mean_baseline, fwd_usingMEG=fwd_usingMEG, fwd_usingEEG=fwd_usingEEG,
        stc_t_smooth=ictal_stc, modality=modality, connectivity=connectivity,
        uuid='{:.2f}-{:.2f}'.format(from_t, to_t), n_jobs=1)
    # labels_fol = op.join(modality_fol, 'clusters', '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
    #     subject, inverse_method, modality, utils.namebase(ictal_clip)))
    # os.rename(labels_fol, '{}_{}_{}'.format(labels_fol, from_t, to_t))


def _calc_ictal_and_baseline_parallel(p):
    (subject, clip_fname, inverse_method, modality, seizure_times, mean_baseline, stcs_fol, reverse, set_t_as_val,\
     n_jobs) = p
    # print('Calculating accumulate_stc for {}'.format(utils.namebase(clip_fname)))
    stc_name = op.join(stcs_fol, '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
        subject, inverse_method, modality, utils.namebase(clip_fname)))
    if not stc_exist(stc_name):
        raise Exception('Cannot find stc_name!')
    stc = mne.read_source_estimate('{}-rh.stc'.format(stc_name))
    # t_from, t_to = stc.time_as_index(seizure_times[0])[0], stc.time_as_index(seizure_times[1])[0]
    t_from, t_to = [stc.time_as_index(seizure_times[k])[0] for k in range(2)]
    # mean_baseline_quick = np.median(np.max(stc.data[:, :stc.time_as_index(0)[0]], axis=0).squeeze())
    # mean_baseline = calc_baseline_mean(subject, stc, n_jobs)
    modality_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality))
    labels_fol = op.join(modality_fol, 'clusters', '{}-epilepsy-{}-{}-{}-amplitude-zvals-{:.2f}-{:.2f}'.format(
        subject, inverse_method, modality, utils.namebase(clip_fname), seizure_times[0], seizure_times[1]))
    lookup_fname = op.join(labels_fol, 'vertices_lookup.pkl')
    if set_t_as_val and not op.isfile(lookup_fname):
        raise Exception('Can\'t find labels lookup! {}'.format(lookup_fname))
    ictal_stc, labels_times = meg.accumulate_stc(
        subject, stc, t_from, t_to, mean_baseline, lookup_fname, reverse=reverse, set_t_as_val=set_t_as_val, n_jobs=n_jobs)
    return clip_fname, stc_name, ictal_stc, mean_baseline, labels_times,


def calc_baseline_mean(subject, stc, n_jobs):
    baseline_stc = stc.copy()
    baseline_stc.crop(baseline_stc.tmin, 0)
    # baseline_stc_mean = baseline_stc.mean()
    # baseline_stc_t = meg.create_stc_t(baseline_stc, 0, subject)
    baseline_stc_smooth = meg.calc_stc_for_all_vertices(baseline_stc, subject, subject, n_jobs)
    # return np.median(np.median(baseline_stc_smooth.data, axis=0).squeeze())
    return np.median(np.max(baseline_stc_smooth.data, axis=0).squeeze())


def calc_func_rois_vertives_lookup(
        subject, ictal_clips, modality, inverse_method, seizure_times, windows_length, windows_shift):
    modality_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality))
    windows = calc_windows(seizure_times, windows_length, windows_shift)
    labels_fols = []
    for ictal_fname in ictal_clips:
        for from_t, to_t in windows:
            labels_fol = op.join(modality_fol, 'clusters', '{}-epilepsy-{}-{}-{}-amplitude-zvals-{:.2f}-{:.2f}'.format(
                subject, inverse_method, modality, utils.namebase(ictal_fname), from_t, to_t))
            labels_fols.append(labels_fol)
    utils.run_parallel(_calc_func_rois_vertives_lookup_parallel, labels_fols, n_jobs)


def _calc_func_rois_vertives_lookup_parallel(labels_fol):
    lookup_fname = op.join(labels_fol, 'vertices_lookup.pkl')
    labels = [mne.read_label(label_fname) for label_fname in glob.glob(op.join(labels_fol, '*.label'))]
    lookup = {'rh': {}, 'lh': {}}
    for label in labels:
        for vertice in label.vertices:
            lookup[label.hemi][vertice] = label.name
    print('Saving {}'.format(lookup_fname))
    utils.save(lookup, lookup_fname)


def calc_rois_connectivity(
        subject, clips, modality, inverse_method, min_order=1, max_order=20, crop_times=(-0.5, 1),
        onset_time=2, windows_length=0.1, windows_shift=0.05, overwrite=False, n_jobs=4):
    # check_connectivity_labels(clips['ictal'], modality, inverse_method, n_jobs=n_jobs)
    windows_length *= 1000
    windows_shift *= 1000
    baseline_epochs_fname = op.join(MMVT_DIR, subject, meg.modality_fol(modality), 'baseline-epo.fif')
    baseline_epochs = epi_utils.combine_windows_into_epochs(clips['baseline'], baseline_epochs_fname)
    params = [(subject, clip_fname, baseline_epochs, modality, inverse_method, min_order, max_order, crop_times,
               onset_time, windows_length, windows_shift, overwrite, 1 if n_jobs > 1 else n_jobs)
              for clip_fname in clips['ictal']]
    utils.run_parallel(calc_clip_rois_connectivity, params, n_jobs)


def check_connectivity_labels(ictal_clips, modality, inverse_method, n_jobs=1):
    # Check if can calc the labels info (if not, we want to know now...)
    for clip_fname in ictal_clips:
        print('Checking {} labels'.format(utils.namebase(clip_fname)))
        labels_fol = op.join(
            MMVT_DIR, subject, meg.modality_fol(modality), 'clusters', '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
                subject, inverse_method, modality, utils.namebase(clip_fname)))
        labels = lu.read_labels_files(subject, labels_fol, n_jobs=n_jobs)
        if len(labels) == 0:
            raise Exception('No labels!')
        else:
            print('{} labels'.format(len(labels)))
        connectivity.calc_lables_info(subject, labels=labels)


def calc_clip_rois_connectivity(p):
    (subject, clip_fname, baseline_epochs, modality, inverse_method, min_order, max_order, crop_times, onset_time,
     windows_length, windows_shift, overwrite, n_jobs) = p
    clusters_fol = op.join(MMVT_DIR, subject, meg.modality_fol(modality), 'clusters')
    fwd_usingMEG, fwd_usingEEG = meg.get_fwd_flags(modality)
    bands = {'all': [None, None]}
    crop_times = [t + onset_time for t in crop_times]
    labels_fol = op.join(
        clusters_fol, '{}-epilepsy-{}-{}-{}-amplitude-zvals'.format(
            subject, inverse_method, modality, utils.namebase(clip_fname)))
    use_functional_rois_atlas = False
    if use_functional_rois_atlas:
        labels = lu.read_labels_files(subject, labels_fol, n_jobs=n_jobs)
        # for connectivity we need shorter names
        labels = epi_utils.shorten_labels_names(labels)
        atlas = utils.namebase(clip_fname)
    else:
        labels = lu.read_labels(subject, SUBJECTS_DIR, 'laus125')
        atlas = 'laus125'  # utils.namebase(clip_fname)
    ictal_evoked = mne.read_evokeds(clip_fname)[0]
    for clip, cond in zip([ictal_evoked, baseline_epochs],
                          [utils.namebase(clip_fname), '{}_baseline'.format(utils.namebase(clip_fname))]):
        meg.calc_labels_connectivity(
            subject, atlas, {cond: 1}, subjects_dir=SUBJECTS_DIR, mmvt_dir=MMVT_DIR, inverse_method=inverse_method,
            pick_ori='normal', fwd_usingMEG=fwd_usingMEG, fwd_usingEEG=fwd_usingEEG,
            con_method='gc', overwrite_connectivity=overwrite, crops_times=crop_times,
            epochs=clip, bands=bands, con_indentifer='func_rois', labels=labels,
            min_order=min_order, max_order=max_order, downsample=2, windows_length=windows_length,
            windows_shift=windows_shift, n_jobs=n_jobs)


def normalize_connectivity(subject, ictals_clips, modality, divide_by_baseline_std, threshold,
                           reduce_to_3d, overwrite=False, n_jobs=6):
    connectivity_template = connectivity.get_output_fname(subject, 'gc', modality, 'mean_flip', 'all_{}_func_rois')
    for clip_fname in ictals_clips:
        clip_name = utils.namebase(clip_fname)
        output_fname = '{}_zvals.npz'.format(connectivity_template.format(clip_name)[:-4])
        con_ictal_fname = connectivity_template.format(clip_name)
        con_baseline_fname = connectivity_template.format('{}_baseline'.format(clip_name))
        if not op.isfile(con_ictal_fname) or not op.isfile(con_baseline_fname):
            for fname in [f for f in [con_ictal_fname, con_baseline_fname] if not op.isfile(f)]:
                print('{} is missing!'.format(fname))
            continue
        print('normalize_connectivity: {}:'.format(clip_name))
        d_ictal = utils.Bag(np.load(con_ictal_fname, allow_pickle=True))
        d_baseline = utils.Bag(np.load(con_baseline_fname, allow_pickle=True))
        if reduce_to_3d:
            d_ictal.con_values = connectivity.find_best_ord(d_ictal.con_values, False)
            d_ictal.con_values2 = connectivity.find_best_ord(d_ictal.con_values2, False)
            d_baseline.con_values = connectivity.find_best_ord(d_baseline.con_values, False)
            d_baseline.con_values2 = connectivity.find_best_ord(d_baseline.con_values2, False)
        d_ictal.con_values = epi_utils.norm_values(
            d_baseline.con_values, d_ictal.con_values, divide_by_baseline_std, threshold, True)
        if 'con_values2' in d_baseline:
            d_ictal.con_values2 = epi_utils.norm_values(
                d_baseline.con_values2, d_ictal.con_values2, divide_by_baseline_std, threshold, True)
        print('Saving norm connectivity in {}'.format(output_fname))
        np.savez(output_fname, **d_ictal)


def plot_connectivity(subject, clips_dict, modality, inverse_method):
    diploes_rois_output_fname = op.join(MMVT_DIR, subject, 'meg', 'dipoles_rois.pkl')
    if op.isfile(diploes_rois_output_fname):
        diploes_rois = utils.load(diploes_rois_output_fname)
    else:
        diploes_rois= None
    nodes_names = []
    for clip_fname in clips_dict['ictal']:
        if False: #diploes_rois is not None:
            diploes = [k for k in diploes_rois.keys() if k.startswith(utils.namebase(clip_fname))]
            if len(diploes) == 0 and op.isfile(diploes_rois_output_fname):
                print('No dipoles found for {}'.format(utils.namebase(clip_fname)))
                continue
            dipole = sorted(diploes)[0]
            nodes_names = diploes_rois[dipole]['cortical_rois']
            nodes_names = list(set(['{}-{}'.format(label.split('_')[0], label[-2:]) for label in nodes_names]))
        plots.plot_connectivity(
            subject, utils.namebase(clip_fname), modality, 120, 'gc', cond_name='', nodes_names=nodes_names,
            nodes_names_includes_hemi=True, stc_subfolder='ictal-{}-zvals-stcs'.format(inverse_method),
            bands={'all':[1, 120]}, stc_downsample=1, stc_name='{}-epilepsy-{}-{}-{}-amplitude-zvals-rh.stc'.format(
                subject, inverse_method, modality, utils.namebase(clip_fname)),
            con_threshold=0)


def delete_morphing_maps(subject):
    # If there is a problem with smoothing the surfaces, you should delete the morphing maps first
    for morph_fname in glob.glob(op.join(SUBJECTS_DIR, 'morph-maps', '*{}*-morph.fif'.format(subject))):
        print('Deleting {}'.format(utils.namebase(morph_fname)))
        utils.delete_file(morph_fname)


def pre_processing(subject, modality, atlas, empty_fname, overwrite=False, n_jobs=4):
    # 0.0) calc fwd inv
    calc_fwd_inv(subject, run_num, modality, raw_fname, empty_fname, bad_channels, overwrite, n_jobs)
    # 0.1) If there is a problem with smoothing the surfaces, you should delete the morphing maps first
    # delete_morphing_maps(subject)
    # 0.2) Finds the trans file
    trans_file = meg.find_trans_file(subject=subject)
    # 0.3) Make sure we have a morph map, and if not, create it here, and not in the parallel function
    mne.surface.read_morph_map(subject, subject, subjects_dir=SUBJECTS_DIR)
    # 0.4) Make sure the label exist, if not, create them
    anat.create_annotation(subject, atlas, n_jobs=n_jobs)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
    if len(labels) == 0:
        raise Exception('No {} labels!'.format(atlas))


def calc_fwd_inv(subject, run_num, modality, raw_fname, empty_fname, bad_channels, overwrite=False, n_jobs=4):
    pipeline.calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels,
                 overwrite_inv=overwrite, overwrite_fwd=overwrite, n_jobs=n_jobs)
    pipeline.check_inv_fwd(subject, modality, run_num)


def main(subject, run_num, clips_dict, raw_fname, empty_fname, bad_channels, modality, inverse_method, downsample_r,
         seizure_times, windows_length, windows_shift, mean_baseline, atlas, min_cluster_size, min_order, max_order,
         con_windows_length, con_windows_shift, con_crop_times, onset_time, overwrite=False, n_jobs=4):
    # pre_processing(subject, modality, atlas, empty_fname, overwrite, n_jobs)
    # calc_stcs(subject, modality, clips_dict, inverse_method, downsample_r, overwrite=overwrite, n_jobs=n_jobs)
    # calc_stc_zvals(subject, modality, clips_dict['ictal'], inverse_method, overwrite=True, n_jobs=n_jobs)
    # find_functional_rois(
    #     subject, clips_dict['ictal'], modality, seizure_times, atlas, min_cluster_size,
    #     inverse_method, mean_baseline, windows_length, windows_shift, overwrite=True, n_jobs=n_jobs)
    # calc_func_rois_vertives_lookup(
    #     subject, clips_dict['ictal'], modality, inverse_method, seizure_times, windows_length, windows_shift)
    # calc_accumulate_stc_as_time(
    #     subject, clips_dict['ictal'], modality, seizure_times, windows_length, windows_shift, mean_baseline,
    #     inverse_method, n_jobs)
    calc_rois_connectivity(
        subject, clips_dict, modality, inverse_method, min_order, max_order, con_crop_times, onset_time,
        windows_length, windows_shift, overwrite=True, n_jobs=n_jobs)
    # # normalize_connectivity(
    #     subject, clips_dict['ictal'], modality, divide_by_baseline_std=False,
    #     threshold=0.5, reduce_to_3d=True, overwrite=False, n_jobs=n_jobs)
    # plot_connectivity(subject, clips_dict, modality, inverse_method)
    pass


if __name__ == '__main__':
    n_jobs = utils.get_n_jobs(40)
    n_jobs = n_jobs if n_jobs >= 1 else 4
    print('{} jobs'.format(n_jobs))
    fif_files, clips_dict = [], {}
    run_num = 3

    subject, remote_subject_dir, meg_fol, bad_channels, raw_fname, empty_room_fname = init_nmr01391()
    for subfol in ['baseline', 'ictal']:
        files = glob.glob(op.join(meg_fol, subfol, 'run{}_*.fif'.format(run_num)))
        fif_files += files
        clips_dict[subfol] = files

    main(subject, run_num, clips_dict, raw_fname, empty_room_fname, bad_channels, modality='meg',inverse_method='MNE',
         downsample_r=2, seizure_times=(-0.2, .5), windows_length=0.1, windows_shift=0.05, mean_baseline=10,
         atlas='laus125', min_cluster_size=50, min_order=1, max_order=20, con_windows_length=100, con_windows_shift=10,
         con_crop_times=(-0.2, 0.5), onset_time=2, overwrite=True, n_jobs=n_jobs)