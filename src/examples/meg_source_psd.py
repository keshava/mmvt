import os.path as op
import os
import glob
import numpy as np
import scipy.stats
from src.utils import utils
from src.preproc import anatomy as anat
from src.preproc import meg
from src.preproc import fMRI as fmri
import mne
from tqdm import tqdm
from collections import defaultdict

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')

MSIT_CONDS = ['Congruent', 'Incongruent']
COND_CON, COND_INC = range(2)


def calc_meg_source_psd(args):
    from src.misc import meg_buddy
    bad_subjects = ['hc004', 'hc012', 'hc029']
    subjects = args.subject
    for subject in subjects:
        if subject in bad_subjects:
            continue
        args.subject = subject
        local_raw_fname = op.join(MEG_DIR, args.task, subject, args.raw_template.format(
            subject=subject, task=args.task))
        remote_raw_fname = op.join(args.remote_root_dir, 'raw_preprocessed', subject, args.raw_template.format(
            subject=subject, task=args.task))
        if not op.isfile(remote_raw_fname) and not args.ignore:
            print('Can\'t find remote raw file! {}'.format(remote_raw_fname))
            continue
        if op.isfile(local_raw_fname):
            os.remove(local_raw_fname)
        utils.make_link(remote_raw_fname, local_raw_fname)
        if not op.islink(local_raw_fname) and not args.ignore:
            print('Can\'t create a link to the remote raw!')
            continue
        inv_fname = op.join(MEG_DIR, args.task, subject, args.inv_template.format(subject=subject, task=args.task))

        _args = meg.read_cmd_args(dict(
            subject=subject, mri_subject=subject,
            function='make_forward_solution,calc_inverse_operator',
            task='MSIT', data_per_task=True,
            fmin=1, fmax=120,
            raw_fname=local_raw_fname, inv_fname=inv_fname,
            remote_subject_dir=args.remote_subject_dir,
            n_jobs=args.n_jobs
        ))
        ret = meg.call_main(_args)
        if not ret[subject].get('calc_inverse_operator', True):
            continue

        # Load the eopchs and calc source power spectrum for both conditions.
        # The epochs are being splitted first
        data = meg_buddy.get_data(subject, tasks=['MSIT'], modalities=['MEG'])['MSIT']['MEG']
        if subject not in data:
            print('{} not in msit_data!'.format(subject))
            continue
        data = data[subject]
        subject_epochs = data._load_epochs('Onset', ar=True)
        indices = data._get_indices(subject_epochs, 'Condition', MSIT_CONDS)
        for cond in MSIT_CONDS:
            epochs = subject_epochs[indices[cond]]
            meg.calc_source_power_spectrum(
                subject, cond.lower(), epochs=epochs, max_epochs_num=50, inv_fname=inv_fname, n_jobs=args.n_jobs)


def calc_source_ttest(args):
    subjects = args.subject
    for subject in subjects:
        args.subject = subject
        fol = op.join(MMVT_DIR, subject, 'meg')
        output_fname = op.join(fol, 'dSPM_mean_flip_vertices_power_spectrum_stat')
        if utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}')):
            print('{} already exist')
            continue
        file_name = '{cond}_dSPM_mean_flip_vertices_power_spectrum.pkl'
        if not all([op.isfile(op.join(fol, file_name.format(cond=cond.lower())))
                for cond in MSIT_CONDS]):
            continue
        vertices_data = {}
        try:
            for cond in MSIT_CONDS:
                vertices_data[cond], freqs = utils.load(op.join(fol, file_name.format(cond=cond.lower())))
        except:
            print('Can\'t read {}'.format(file_name.format(cond=cond.lower())))
            continue
        pvals, vertno = defaultdict(list), {}
        for hemi in utils.HEMIS:
            vertices_inds = {}
            for cond in MSIT_CONDS:
                vertices_inds[cond] = np.array(sorted(list(vertices_data[MSIT_CONDS[0]][hemi].keys())))
            if not np.all(vertices_inds[MSIT_CONDS[0]] == vertices_inds[MSIT_CONDS[1]]):
                raise Exception('Not the same vertices!')
            vertno[hemi] = vertices_inds[MSIT_CONDS[0]]
            for vert in tqdm(vertices_data[MSIT_CONDS[0]][hemi].keys()):
                x = [vertices_data[cond][hemi][vert] for cond in MSIT_CONDS]
                t, pval = scipy.stats.ttest_ind(x[0], x[1], equal_var=False)
                pvals[hemi].append(-np.log10(pval))

        data = np.concatenate([pvals['lh'], pvals['rh']])
        vertices = [vertno['lh'], vertno['rh']]
        stc_pvals = mne.SourceEstimate(data, vertices, freqs[0], freqs[1] - freqs[0], subject=subject)
        print('Writing to {}'.format(output_fname))
        stc_pvals.save(output_fname)


def calc_pvals_clusters(args):
    utils.run_parallel(_calc_pvals_clusters, args.subject, args.n_jobs)


def _calc_pvals_clusters(subject):
    stc_name = 'dSPM_mean_flip_vertices_power_spectrum_stat'
    if not utils.both_hemi_files_exist(
            op.join(MMVT_DIR, subject, 'meg', '{}-{}.stc'.format(stc_name, '{hemi}'))):
        print('{}: Can\'t find {}!'.format(subject, stc_name))
        return
    args.subject = subject
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'meg', 'clusters'))
    utils.delete_folder_files(clusters_root_fol)
    _args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        function='find_functional_rois_in_stc',
        stc_name=stc_name,
        threshold=-np.log10(0.05), threshold_is_precentile=False,
        extract_time_series_for_clusters=True,
        n_jobs=args.n_jobs
    ))
    meg.call_main(_args)


def morph_stcs_pvals(args):
    utils.run_parallel(_morph_stcs_pvals, args.subject, args.n_jobs)


def _morph_stcs_pvals(subject):
    fol = utils.make_dir(op.join(MMVT_DIR, args.morph_target, 'meg', 'morphed'))
    output_fname = op.join(fol, '{}_dSPM_mean_flip_power_spectrum_stat'.format(subject))
    if utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}')) and not args.overwrite:
        return
    stc_fname = op.join(MMVT_DIR, subject, 'meg', 'dSPM_mean_flip_vertices_power_spectrum_stat-{}.stc'.format('{hemi}'))
    if not utils.both_hemi_files_exist(stc_fname):
        return
    stc = mne.read_source_estimate(stc_fname.format(hemi='lh'))
    stc_morphed = mne.morph_data(subject, args.morph_target, stc, grade=None, n_jobs=args.n_jobs)
    print('Saving {}'.format(output_fname))
    stc_morphed.save(output_fname)


def average_stc_pvals(args):
    fol = utils.make_dir(op.join(MMVT_DIR, args.morph_target, 'meg'))
    stc_mean = None
    stcs_num = 0
    pval_lower_limit = 5
    stc_name = 'dSPM_mean_flip_power_spectrum_stat'
    if utils.both_hemi_files_exist('{}-{}.stc'.format(op.join(fol, stc_name), '{hemi}')) and not args.overwrite:
        print('{} already exist'.format(op.join(fol, stc_name)))
        return
    stcs_fnames = glob.glob(op.join(fol, 'morphed',  '*_{}-lh.stc'.format(stc_name)))
    for stc_fname in tqdm(stcs_fnames):
        stc = mne.read_source_estimate(stc_fname, args.morph_target)
        stc_max = utils.max_stc(stc)
        if stc_max > pval_lower_limit:
            print('{}, max: {:.2f}'.format(stc_fname, stc_max))
            stc_mean = stc if stc_mean is None else stc_mean + stc
            stcs_num += 1
    stc_mean /= stcs_num
    print('Writing avg stc to {}'.format(op.join(fol, stc_name)))
    stc_mean.save(op.join(fol, stc_name))


# def find_meg_psd_clusters(args):
#     subjects = args.subject
#     stc_name = 'all_dSPM_mean_flip_high_gamma_power'
#     for subject in subjects:
#         if not utils.both_hemi_files_exist(
#                 op.join(MMVT_DIR, subject, 'meg', '{}-{}.stc'.format(stc_name, '{hemi}'))):
#             print('{}: Can\'t find {}!'.format(subject, stc_name))
#             continue
#         args.subject = subject
#         _args = meg.read_cmd_args(dict(
#             subject=subject, mri_subject=subject,
#             function='find_functional_rois_in_stc',
#             stc_name=stc_name,
#             threshold=95, threshold_is_precentile=True,
#             extract_time_series_for_clusters=False,
#             n_jobs=args.n_jobs
#         ))
#         meg.call_main(_args)
#
#
# def morph_meg_powers(args):
#     utils.run_parallel(_morph_meg_powers, args.subject, args.n_jobs)
#
#
# def _morph_meg_powers(subject):
#     fol = utils.make_dir(op.join(MMVT_DIR, args.morph_target, 'meg', 'morphed'))
#     bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
#     for band, stc_band in bands.items():
#         output_fname = op.join(fol, '{}_{}'.format(subject, band))
#         if utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}')) and not args.overwrite:
#             continue
#         stc_fname = op.join(MMVT_DIR, subject, 'meg', 'all_dSPM_mean_flip_{}_power-{}.stc'.format(
#             band, '{hemi}'))
#         if not utils.both_hemi_files_exist(stc_fname):
#             continue
#         stc = mne.read_source_estimate(stc_fname.format(hemi='lh'))
#         stc_morphed = mne.morph_data(subject, args.morph_target, stc, grade=None, n_jobs=args.n_jobs)
#         print('Saving {}'.format(output_fname))
#         stc_morphed.save(output_fname)
#
#
# def normalize_meg_source_psd(args):
#     utils.run_parallel(_normalize_meg_source_psd, args.subject, args.n_jobs)
#
#
# def _normalize_meg_source_psd(subject):
#     normalize_to_one = True
#     bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
#     fol = utils.make_dir(op.join(MMVT_DIR, args.morph_target, 'meg', 'morphed'))
#     subject_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'meg'))
#     for band, stc_band in bands.items():
#         output_fname = op.join(fol, '{}_{}_norm'.format(subject, band))
#         if utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}')) and not args.overwrite:
#             continue
#         file_name = op.join(fol, '{}_{}'.format(subject, band))
#         if not utils.both_hemi_files_exist('{}-{}.stc'.format(file_name, '{hemi}')):
#             print('{} is missing!'.format(file_name))
#             continue
#         if not normalize_to_one:
#             baseline_file_name = op.join(subject_fol, 'MSIT_dSPM_baseline_{}'.format(band))
#             if not utils.both_hemi_files_exist('{}-{}.stc'.format(baseline_file_name, '{hemi}')):
#                 print('{} is missing!'.format(baseline_file_name))
#                 continue
#         psd_stc = mne.read_source_estimate('{}-rh.stc'.format(file_name), subject)
#         if normalize_to_one:
#             max_psd = utils.max_stc(psd_stc)
#             max_data = np.concatenate(
#                 [np.ones((len(psd_stc.lh_data), 1)) * max_psd, np.ones((len(psd_stc.rh_data), 1)) * max_psd])
#             baseline_stc = mne.SourceEstimate(max_data, psd_stc.vertices, 0, 0, subject=subject)
#         else:
#             baseline_stc = mne.read_source_estimate('{}-rh.stc'.format(baseline_file_name), subject)
#         stc_band_norm_power = meg.normalize_stc(subject, psd_stc, baseline_stc)
#         print('Saving {}'.format(output_fname))
#         stc_band_norm_power.save(output_fname)
#
#
# def average_meg_powers(args):
#     bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
#     fol = utils.make_dir(op.join(MMVT_DIR, args.morph_target, 'meg'))
#     power_upper_limit = 1e4
#     powers = defaultdict(list)
#     stc_mean = {}
#     stcs_num = {}
#     for band, stc_band in bands.items():
#         stc_mean[band] = None
#         stcs_num[band] = 0
#         stcs_fnames = glob.glob(op.join(fol, 'morphed',  '*_{}_norm-lh.stc'.format(band)))
#         for stc_fname in stcs_fnames:
#             stc = mne.read_source_estimate(stc_fname, args.morph_target)
#             min_stc, max_stc = utils.calc_min_max_stc(stc)
#             if max_stc > power_upper_limit:
#                 continue
#             powers[band].append(utils.calc_mean_stc(stc))
#             stc_mean[band] = stc if stc_mean[band] is None else stc_mean[band] + stc
#             stcs_num[band] += 1
#     for band, stc_band in bands.items():
#         stc_mean[band] /= stcs_num[band]
#         stc_mean[band].save(op.join(fol, 'MSIT_mean_{}'.format(band)))
#         print('{} {}+-{}'.format(band, np.mean(powers[band]), np.std(powers[band])))


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    from src.utils import preproc_utils as pu

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='')
    parser.add_argument('-a', '--atlas', required=False, default='laus125')
    parser.add_argument('-t', '--task', required=False, default='MSIT')
    parser.add_argument('-f', '--function', help='function name', required=False, default='analyze_meg')

    parser.add_argument('--remote_root_dir', required=False,
                        default='/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/')
    meg_dirs = ['/home/npeled/meg/{task}',
                '/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/epochs']
    meg_dir = [d for d in meg_dirs if op.isdir(d.format(task='MSIT'))][0]
    parser.add_argument('--meg_dir', required=False, default=meg_dir)
    remote_subject_dirs = ['/autofs/space/lilli_001/users/DARPA-Recons/',
                           '/home/npeled/subjects']
    remote_subject_dir = [op.join(d, '{subject}') for d in remote_subject_dirs if op.isdir(d)][0]
    parser.add_argument('--remote_subject_dir', required=False, default=remote_subject_dir)
    parser.add_argument('--remote_meg_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg')
    parser.add_argument('--epo_template', required=False, default='{subject}_{task}_meg_Onset_ar-epo.fif')
    parser.add_argument('--raw_template', required=False, default='{subject}_{task}_meg_ica-raw.fif')
    parser.add_argument('--inv_template', required=False, default='{subject}_{task}_Onset-inv.fif')
    parser.add_argument('--morph_target', required=False, default='fsaverage5')
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--ignore', required=False, default=False, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.subject = pu.decode_subjects(args.subject, remote_subject_dir=args.remote_subject_dir)
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
    print('Done!')