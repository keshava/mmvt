import os.path as op
import os
import time
import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
from collections import defaultdict
from itertools import cycle, product

import warnings
# warnings.filterwarnings('error')


from src.utils import utils
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu
from src.utils import labels_utils as lu
from src.preproc import fMRI
from src.preproc import anatomy as anat

FS_ROOT = '/autofs/space/nihilus_001/CICS/recons'
FS_BASE_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_recons'
FS_BASE_6_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_6month_recons'
HOME_FOL = '/autofs/space/nihilus_001/CICS/users/noam/CICS/'
RESULTS_FOL = [d for d in ['/autofs/space/nihilus_001/CICS/users/noam/results/', 'C:\\Users\\peled\\CICS']
               if op.isdir(d)][0]
SCAN, RESCAN = 'scan', 'rescan'
SCAN_RESCAN = [SCAN, RESCAN]

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
print('Setting SUBJECTS_DIR to {}'.format(SUBJECTS_DIR))
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
# setenv SUBJECTS_DIR /autofs/space/nihilus_001/CICS/users/noam/mmvt_root/subjects

mri_robust_register = 'mri_robust_register --mov "{source_fname}" --dst "{target_fname}" --lta "{lta_fname}" ' + \
                      '--satit --mapmov "{output_fname}" --cost {cost_function}'
bbregister = 'bbregister --s {subject} --mov "{source_fname}" --lta "{lta_fname}" --t1 --o "{output_fname}"'
             # '--tmp "{tmp_fol}"'
register_using_lta = 'mri_convert -at "{lta_fname}" "{source_fname}" "{output_fname}"'
register_using_inverse_lta = 'mri_convert -ait "{lta_fname}" "{source_fname}" "{output_fname}"'
mri_compute_volume_fractions = 'mri_compute_volume_fractions --o "{output_fname}" --regheader {subject} "{target_fname}"'
mri_compute_volume_fractions_reg = 'mri_compute_volume_fractions --o "{output_fname}" --reg  "{reg_fname}"'
save_registration_figure = 'freeview -v "{target_fname}":visible=0:name=orig.mgz {source_fname}:name=Control.nii:reg={lta_fname} --surface {white_lh_fname}:edgecolor=yellow --surface {white_rh_fname}:edgecolor=yellow --ss {output_fig_fname}'


def get_subject_fs_folder(subject, scan_rescan, month):
    # base_6_12_str = '' if base_6_12 == '0' else base_6_12
    # scan_rescan_str = base_6_12_str if scan_rescan == SCAN else '{}B'.format(base_6_12_str)
    # if scan_rescan_str != '':
    #     scan_rescan_str += '_'
    # return '{0}_{1}recon.long.{0}-base'.format(subject, scan_rescan_str)
    subject_fol = '{}_{}_recon'.format(
        subject, month if month != '0' else '', 'B' if scan_rescan == RESCAN else '')
    subject_fol = subject_fol.replace('__', '_')
    return subject_fol



def input_files_exist(input_fnames):
    return all([op.isfile(fname) for fname in input_fnames])


def output_exists(output_fnames, overwrite):
    if all_files_exits(output_fnames):
        try:
            # Try to load the file to check if it's not currupted
            for output_fname in output_fnames:
                file_type = utils.file_type(output_fname)
                if file_type == 'pkl':
                    utils.load(output_fname)
                elif file_type in ['mgz', 'nii', 'nii.gz', 'mgh']:
                    nib.load(output_fname)
        except:
            print('*** Cannot load the output files!')
            print(output_fnames)
            return False
    return False if overwrite else all_files_exits(output_fnames)


def check_if_should_run_function(input_fnames, output_fnames, subject_fol, overwrite, func_name, verbose=True):
    do_input_files_exist = input_files_exist(input_fnames)
    do_output_exists = output_exists(output_fnames, overwrite)
    if verbose and not do_input_files_exist:
        print('{}: input files does not exist! ({})'.format(func_name, subject_fol))
    if verbose and do_output_exists:
        print('{}: output exist, continue ({})'.format(func_name, subject_fol))
    return input_files_exist and not do_output_exists


def set_subject(subject, scan_rescan, month):
    # subject = subject if scan_rescan == SCAN else '{}_rescan'.format(subject)
    full_subject_name = get_full_subject_name(subject, month, scan_rescan)
    os.environ['SUBJECT'] = full_subject_name
    return full_subject_name


def get_full_subject_name(subject, month, scan_rescan=''):
    return '{}{}{}'.format(
        subject, '_rescan' if scan_rescan == RESCAN else '',
        '_{}'.format(month) if month != '0' else '')


def check_input_output(input_fnames, output_fnames, subject_fol, overwrite, func_name, verbose):
    if not input_files_exist(input_fnames):
        if verbose:
            print('{}: input files does not exist! ({})'.format(func_name, subject_fol))
        return False, False
    if output_exists(output_fnames, overwrite):
        return False, True
    return True, True


def all_files_exits(output_fnames):
    if isinstance(output_fnames, dict):
        output_fnames = list(output_fnames.values())
    return all([op.isfile(fname) for fname in output_fnames])


def register_cbf_to_t1(subject, subject_fol, scan_rescan, month, overwrite=False, print_only=False, check_if_should_run=False,
                       verbose=True):
    output_fname = op.join(subject_fol, 'Control_to_T1.nii')
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    source_fname = op.join(subject_fol, 'Control.nii')
    target_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan, month), 'mri', 'T1.mgz')
    do_run, ret = cics_checks(
        [source_fname, target_fname], [output_fname, lta_fname], subject_fol, check_if_should_run, overwrite, verbose)
    if not do_run:
        return ret
    set_subject(subject, scan_rescan, month)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # bbregister --s {subject} --mov "{source_fname}" --lta "{lta_fname}" --t1 --o "{output_fname}"
    rs(bbregister)
    return all([op.isfile(fname) for fname in [output_fname, lta_fname]])


def check_cbf_to_t1_registeration(subject, site, scan_rescan, print_only=False):
    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    subject_fs_fol = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan))
    orig_fname = op.join(subject_fs_fol, 'mri', 'orig.mgz')
    control_fname = op.join(subject_fol, 'Control.nii')
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    lh_white = op.join(subject_fs_fol, 'surf', 'lh.white')
    rh_white = op.join(subject_fs_fol, 'surf', 'rh.white')
    cmd = 'freeview -transform-volume -viewport cor -v "{orig_fname}":visible=0:name=orig.mgz ' + \
        '"{control_fname}":name=Control.nii:reg="{lta_fname}" --surface ' + \
        '"{lh_white}":edgecolor=yellow --surface "{rh_white}":edgecolor=yellow'
    utils.partial_run_script(locals(), print_only=print_only)(cmd)


def save_registration_results():
    '''
    freeview -v {target_fname}:visible=0:name=orig.mgz {source_fname}:name=Control.nii:reg={lta_fname} --surface {white_lh_fname}:edgecolor=yellow --surface {white_rh_fname}:edgecolor=yellow --ss {output_fig_fname}
    '''

    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    target_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.mgz')
    source_fname = op.join(subject_fol, 'Control.nii')
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    white_lh_fname =  op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'surf', 'lh.white')
    white_rh_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'surf', 'rh.white')
    output_fig_fname = op.join(subject_fol, 'control_to_T1.jpg')


def cics_checks(input_fnames, output_fnames, subject_fol, check_if_should_run, overwrite, verbose,
                delete_output_files=False):
    func_name = utils.caller_func()
    if check_if_should_run:
        ret = check_if_should_run_function(
            input_fnames, output_fnames, subject_fol, overwrite, func_name, verbose)
        return False, ret
    if overwrite and delete_output_files:
        for output_fname in output_fnames:
            utils.delete_file(output_fname)
    do_run, ret = check_input_output(
        input_fnames, output_fnames, subject_fol, overwrite, func_name, verbose)
    return do_run, ret


def register_aseg_to_cbf(
        subject, subject_fol, scan_rescan, month, overwrite=False, print_only=True,
        check_if_should_run=False, verbose=True):
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    source_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan, month), 'mri', 'aparc+aseg.mgz')
    output_fname = op.join(subject_fol, 'aparc+aseg_cbf.mgz')
    do_run, ret = cics_checks(
        [lta_fname, source_fname], [output_fname], subject_fol, check_if_should_run, overwrite, verbose)
    if not do_run:
        return ret
    subject = set_subject(subject, scan_rescan, month)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # 'mri_convert -ait "{lta_fname}" "{source_fname}" "{output_fname}"'
    rs(register_using_inverse_lta)
    return op.isfile(output_fname)


def calc_regions_stats(
        subject, subject_fol, scan_rescan, month, low_threshold, high_threshold, cortex_frac_threshold=0.9,
        overwrite=False, do_plot=True, check_if_should_run=False, verbose=True):
    # inputs
    cics_cbf_fname = op.join(subject_fol, 'CBF.nii')
    aparc_aseg_fname = op.join(subject_fol, 'aparc+aseg_cbf.mgz')
    cortex_frac_fname = op.join(subject_fol, 'CBF.cortex.mgz')
    input_fnames = [cics_cbf_fname, aparc_aseg_fname, cortex_frac_fname]
    # outputs
    subject_full_name = get_full_subject_name(subject, month, scan_rescan)
    figures_fol = utils.make_dir(op.join(RESULTS_FOL, subject, 'cbf_hists'))
    stats_output_fname = op.join(
        RESULTS_FOL, subject, '{}_gw_frac_{}_regions_stats.pkl'.format(subject_full_name, cortex_frac_threshold))
    subcortical_stats_output_fname = op.join(
        RESULTS_FOL, subject, '{}_gw_frac_{}_subcortical_stats.pkl'.format(subject_full_name, cortex_frac_threshold))
    all_values_fname = op.join(
        RESULTS_FOL, subject, '{}_gw_frac_{}_regions_values.npy'.format(subject_full_name, cortex_frac_threshold))
    output_fnames = [stats_output_fname, subcortical_stats_output_fname, all_values_fname]
    do_run, ret = cics_checks(
        input_fnames, output_fnames, subject_fol, check_if_should_run, overwrite, verbose)
    if not do_run:
        return ret

    cbf_data = nib.load(cics_cbf_fname).get_data()
    cortex_frac = nib.load(cortex_frac_fname).get_data()
    lut = utils.read_freesurfer_lookup_table(return_dict=True)
    aparc_aseg = nib.load(aparc_aseg_fname).get_data()
    unique_codes = list(range(0, 100)) + list(range(1001, 1036)) + list(range(2001, 2036))
    if overwrite and do_plot:
        utils.delete_folder_files(figures_fol)
    regions_values, regions_means, subcortical_values, subcortical_means, regions_fracs = {}, {}, {}, {}, {}
    all_values = []
    output_str = ''
    for code in tqdm(unique_codes):
        region_name = lut.get(code, None)
        if region_name is None:
            # print('{} not in lut!'.format(code))
            continue
        fig_fname = op.join(figures_fol, '{}.jpg'.format(region_name))
        if op.isfile(fig_fname) and not overwrite:
            continue
        region_values = cbf_data[np.where(aparc_aseg == code)]
        if len(region_values) == 0:
            output_str += 'no voxels for code {}'.format(code)
            continue
        cortex_frac_values = cortex_frac[np.where(aparc_aseg == code)]
        cortex_indices = np.where(cortex_frac_values > cortex_frac_threshold)
        regions_fracs[region_name] = len(cortex_indices[0]) / len(region_values)
        if len(cortex_indices[0]) == 0:
            output_str += '{} has no voxels with cortex_frac > {}!\n'.format(region_name, cortex_frac_threshold)
            continue
        region_values = region_values[cortex_indices]
        # regions_values[region_name] = region_values.copy()
        in_bounderies_indices = np.where((low_threshold < region_values) & (region_values < high_threshold))
        region_values = region_values[in_bounderies_indices]
        if code < 100:
            subcortical_values[region_name] = region_values.copy()
            subcortical_means[region_name] = np.median(subcortical_values[region_name])
        if code >= 1000:
            all_values.extend(region_values.tolist())
            regions_values[region_name] = region_values.copy()
            regions_means[region_name] = np.median(regions_values[region_name])
            if do_plot:
                plt.hist(region_values, bins=40)
                plt.savefig(fig_fname)
                plt.close()
    if verbose:
        print(output_str)
    np.save(all_values_fname, np.array(all_values))
    utils.save((regions_values, regions_means, regions_fracs), stats_output_fname)
    utils.save((subcortical_values, subcortical_means, regions_fracs), subcortical_stats_output_fname)
    return all_files_exits(output_fnames)


def plot_global_data(sites, months, cortex_frac_threshold=0.5, use_subplot=False, overwrite=False):
    global_mean_fname = op.join(RESULTS_FOL, 'global_means_{}.pkl'.format(cortex_frac_threshold))
    if op.isfile(global_mean_fname) and not overwrite:
        cortical_data, hippo_data, site_subjects = utils.load(global_mean_fname)
    else:
        cortical_data, hippo_data, site_subjects = calc_global_means(sites, months, cortex_frac_threshold)
        utils.save((cortical_data, hippo_data, site_subjects), global_mean_fname)
    colors = cycle(utils.get_distinct_colors(len(sites)))
    subjects_ages_dict = get_subjects_ages()
    min_age = min(subjects_ages_dict.values())
    max_age = max(subjects_ages_dict.values())
    titles = {'0':'baseline', '6': '6 months', '12': '1 year'}
    for data, ylabel in zip([cortical_data, hippo_data], ['ASL cortical mean', 'ASL Hippocampus mean']):
        if use_subplot:
            fig, axs = plt.subplots(len(months), sharex=True, sharey=True)#, figsize=(15, 8))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            axs = [ax] * len(months)
        for site, color in zip(sites, colors):
            subjects = site_subjects[site]
            for month, ax in zip(months, axs):
                x = [subjects_ages_dict.get(s, -10) + int(month) / 12 for s in subjects]
                ax.scatter(x, data[site][month][SCAN], c=color, marker='+', label='{} scan'.format(site))
                ax.scatter(x, data[site][month][RESCAN], c=color, marker='x', label='{} rescan'.format(site))
                ax.plot((x, x), (data[site][month][SCAN], data[site][month][RESCAN]), c='black')
                if use_subplot:
                    ax.set_title(titles[month])
                ax.set_xlim((min_age - 5, max_age + 5))
        all_handles, all_labels = plt.gca().get_legend_handles_labels()
        labels, handles = [], []
        for l, h in zip(all_labels, all_handles):
            if l not in labels:
                labels.append(l)
                handles.append(h)
        # fig.legend(handles, labels, loc='lower center', ncol=2, mode='expand')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axs[0].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
        axs[1].set_ylabel(ylabel)
        axs[2].set_xlabel('Age')
        plt.subplots_adjust(left=0.06, bottom=0.09, right=0.79, top=0.94, wspace=0, hspace=0.2)
        # fig.subplots_adjust(right=0.7)
        # fig.tight_layout()
        figure_fname = op.join(RESULTS_FOL, '{}_vs_age_frac_{}.jpg'.format(
            ylabel.replace(' ', '_'), cortex_frac_threshold))
        print('Saving figure in {}'.format(figure_fname))
        # plt.savefig(figure_fname, bbox_inches='tight', dpi=100)
        plt.show()


def calc_global_means(sites, months, cortex_frac_threshold=0.5):
    global_mean_dict, hippocampus_values_dict, site_subjects = {}, {}, {}
    for site in sites:
        global_mean_dict[site] = defaultdict(dict)
        hippocampus_values_dict[site] = defaultdict(dict)
        site_subjects[site] = get_subjects(site)
        for subject in site_subjects[site]:
            for month in months:
                for scan_rescan in SCAN_RESCAN:
                    if scan_rescan not in global_mean_dict[site][month]:
                        global_mean_dict[site][month][scan_rescan] = []
                    if scan_rescan not in hippocampus_values_dict[site][month]:
                        hippocampus_values_dict[site][month][scan_rescan] = []
                    subject_full_name = get_full_subject_name(subject, month, scan_rescan)
                    all_values_fname = op.join(
                        RESULTS_FOL, subject, '{}_gw_frac_{}_regions_values.npy'.format(
                            subject_full_name, cortex_frac_threshold))
                    if op.isfile(all_values_fname):
                        global_mean_dict[site][month][scan_rescan].append(np.load(all_values_fname).mean())
                    else:
                        global_mean_dict[site][month][scan_rescan].append(None)
                    subcortical_stats_input_fname = op.join(
                        RESULTS_FOL, subject, '{}_gw_frac_{}_subcortical_stats.pkl'.format(
                            subject_full_name, cortex_frac_threshold))
                    if op.isfile(all_values_fname):
                        subcortical_values, _, _ = utils.load(subcortical_stats_input_fname)
                        hippocampus_values = subcortical_values.get('Left-Hippocampus', np.array([])).tolist() + \
                                             subcortical_values.get('Right-Hippocampus', np.array([])).tolist()
                        hippocampus_mean = np.mean(hippocampus_values) if len(hippocampus_values) > 0 else None
                        # if hippocampus_mean is None:
                        #     print('hippo is None!', site, subject,month, scan_rescan)
                        hippocampus_values_dict[site][month][scan_rescan].append(hippocampus_mean)
                    else:
                        hippocampus_values_dict[site][month][scan_rescan].append(None)

    return global_mean_dict, hippocampus_values_dict, site_subjects


def get_subjects_ages():
    csv_fname = op.join(HOME_FOL, 'subjects_age.csv')
    if not op.isfile(csv_fname):
        raise Exception('Cannot find csv file for subjects age! ({})'.format(csv_fname))
    return {name: int(age) for name, age in np.genfromtxt(csv_fname, delimiter=',', dtype=np.str, skip_header=1)}


def calc_T1_CNR(subject, subject_fol, scan_rescan, month, print_only=False, overwrite=False, check_if_should_run=False, verbose=True):
    # inputs
    subject_fs_folder = get_subject_fs_folder(subject, scan_rescan, month)
    cortex_label_template = op.join(FS_ROOT, subject_fs_folder, 'label', '{hemi}.cortex.label')
    wg_pct_template = op.join(FS_ROOT, subject_fs_folder, 'surf',  '{hemi}.w-g.pct.mgh')
    input_fnames = [cortex_label_template.format(hemi='rh'), cortex_label_template.format(hemi='lh'),
                    wg_pct_template.format(hemi='rh'), wg_pct_template.format(hemi='lh')]
    # outputs
    sum_template = op.join(subject_fol, 'summary.{}.{}.dat'.format(subject, '{hemi}'))
    output_fnames = [sum_template.format(hemi='rh'), sum_template.format(hemi='lh')]
    do_run, ret = cics_checks(
        input_fnames, output_fnames, subject_fol, check_if_should_run, overwrite, verbose)
    if not do_run:
        return ret

    subject = set_subject(subject, scan_rescan, month)
    for hemi in utils.HEMIS:
        cortex_label_fname = cortex_label_template.format(hemi=hemi)
        wg_pct_fname = wg_pct_template.format(hemi=hemi)
        sum_fname = sum_template.format(hemi=hemi)
        rs = utils.partial_run_script(locals(), print_only=print_only)
        mri_segstats = 'mri_segstats --slabel {subject} {hemi} {cortex_label_fname} ' + \
              '--i {wg_pct_fname} --id 1 --snr --sum {sum_fname}'
        rs(mri_segstats)

    return all_files_exits(output_fnames)


def average_T1_CNR(subjects, site, month, do_plot_hist, overwrite, check_if_should_run=False, verbose=False):
    # input
    input_fnames, input_files_num = [], 0
    for subject, scan_rescan in product(subjects, SCAN_RESCAN):
        subject_fol = get_subject_dir(subject, site, month, scan_rescan)
        input_template = op.join(subject_fol, 'summary.{}.{}.dat'.format(subject, '{hemi}'))
        if utils.both_hemi_files_exist(input_template):
            input_fnames.append(input_template)
    print('{}-{}: {}/{} input files'.format(site, month, len(input_fnames), len(subjects) * 2))
    if len(input_fnames) == 0:
        return False

    # output
    output_fnames = {}
    cics_checks([], output_fnames, subject, check_if_should_run, overwrite, verbose)
    if check_if_should_run:
        return True

    all_snr = []
    for input_template in input_fnames:
        snrs = [np.genfromtxt(input_template.format(hemi=hemi))[-1] for hemi in utils.HEMIS]
        all_snr.append(np.mean(snrs))
    month_str = {'0': 'baseline', '6': '6 months', '12': '1 year'}
    print('{} {}: {:.2f} mean snr ({:.2f} std) ({} subjects)'.format(
        site, month_str[month], np.mean(all_snr), np.std(all_snr), len(all_snr)))
    return True

# def _plot_cbf_histograms_parallel(p):
#     subject, scan_rescan = p
#     input_fname = op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'aparc_aseg_hist.pkl')
#     output_fol = utils.make_dir(op.join(RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan))
#     if not op.isfile(input_fname):
#         return False
#     regions_values = utils.load(input_fname)
#     for region_name, region_values in regions_values.values():
#         fig_fname = op.join(output_fol, '{}.jpg'.format(region_name))
#         plt.hist(region_values, bins=40)
#         plt.savefig(fig_fname)
#         plt.close()


# def print_freeview_cmd(subject, subject_fol):
#     cbf = op.join(subject_fol, 'CBF_to_T1.nii')
#     control = op.join(subject_fol, 'Control_to_T1.nii')
#     t1 = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
#     print('freeview {} {} {}'.format(cbf, control, t1))

def get_labels_data(subject, atlas):
    necessary_files = {'label': ['{}.{}.annot'.format(hemi, atlas) for hemi in utils.HEMIS]}
    remote_subject_dir = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject))
    utils.prepare_subject_folder(necessary_files, subject, remote_subject_dir, SUBJECTS_DIR)


def preproc_anat(subject, month, scan_rescan, overwrite_files=False):
    good_subjects = []
    # for subject in subjects:
    subject_name = '{}{}{}'.format(
        subject, '_rescan' if scan_rescan == RESCAN else '',
        '_{}'.format(month) if month != '0' else '')
    subject_fol = '{}_{}_recon'.format(
        subject, month if month != '0' else '', 'B' if scan_rescan == RESCAN else '')
    subject_fol = subject_fol.replace('__', '_')
    # fs_dir = op.join(FS_BASE_6_ROOT, '{0}_{1}{}recon.long.{0}-base'.format(
    #     subject, '{}_'.format(month) if month != '0' else ''))
    # fs_rescan_dir = op.join(FS_BASE_6_ROOT, '{0}_{1}B_recon.long.{0}-base'.format(
    #     subject, month if month != '0' else ''))
    fs_dir = op.join(FS_ROOT, subject_fol)
    # if not op.isdir(fs_dir):
    #     print('{} does not exist!'.format(fs_dir))
    #     continue
    # if not op.isdir(fs_rescan_dir):
    #     print('{} does not exist!'.format(fs_rescan_dir))
    #     continue
    # # utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    # utils.make_dir(op.join(MMVT_DIR, '{}_rescan'.format(subject), 'fmri'))
    scan_ret = utils.prepare_subject_folder(
        anat.get_necessary_files(), subject_name, fs_dir, SUBJECTS_DIR,
        overwrite_files=overwrite_files, create_links=True)[0]
    return scan_ret
    # rescan_ret = utils.prepare_subject_folder(
    #     anat.get_necessary_files(), rescan_subject_name, fs_rescan_dir, SUBJECTS_DIR,
    #     overwrite_files=overwrite_files)[0]
    # if scan_ret: # and rescan_ret:
    #     good_subjects.append(subject)
    # print('{}/{} good subjects'.format(len(good_subjects), len(subjects)))
    # return good_subjects


@utils.tryit()
def project_cbf_on_cortex(subject, site, scan_rescan, overwrite=False, print_only=False):
    mmvt_subject = subject if scan_rescan == SCAN else '{}_rescan'.format(subject)
    fmri_fol = utils.make_dir(op.join(FMRI_DIR, mmvt_subject))
    mmvt_cbf_fname = op.join(fmri_fol, 'CBF_{}.nii'.format(scan_rescan))
    cics_cbf_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF.nii')
    reg_file = op.join(HOME_FOL, site, subject, scan_rescan, 'control_to_T1.lta')
    if not op.islink(mmvt_cbf_fname) and op.isfile(cics_cbf_fname):
        utils.make_link(cics_cbf_fname, mmvt_cbf_fname)
    if not op.islink(mmvt_cbf_fname):
        print('Cannot file the link to CBF! ({}-{})'.format(cics_cbf_fname, mmvt_cbf_fname))
        return False
    if not op.isfile(reg_file):
        print('Cannot find the registration file! ({})'.format(reg_file))
        return False
    verts = utils.load_surf(subject, MMVT_DIR, SUBJECTS_DIR)
    for hemi in utils.HEMIS:
        surf_output_fname = op.join(FMRI_DIR, subject, 'CBF_{}_{}.mgz'.format(scan_rescan, hemi))
        npy_output_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_CBF_{}_{}.npy'.format(scan_rescan, hemi))
        if op.isfile(npy_output_fname) and not overwrite:
            continue
        fu.project_volume_data(
            cics_cbf_fname, hemi, reg_file=reg_file, smooth_fwhm=0, output_fname=surf_output_fname,
            print_only=print_only)
        if print_only:
            continue
        surf_data = np.squeeze(nib.load(surf_output_fname).get_data())
        np.save(npy_output_fname, surf_data)
        if len(verts[hemi]) != len(surf_data):
            print('*** Wrong number of vertices in {} data! surf vertices ({}) != data vertices ({})'.format(
                hemi, len(verts[hemi]), len(surf_data)))

    # args = fMRI.read_cmd_args(dict(
    #     subject=mmvt_subject,
    #     function='project_volume_to_surface',
    #     fmri_file_template=utils.namebase_with_ext(mmvt_cbf_fname),
    #     overwrite_surf_data=overwrite))
    # fMRI.call_main(args)
    # copy the rescan to the scan folder
    # mmvt_blend/277S0203_rescan/fmri/fmri_CBF_rescan_rh.npy
    # if scan_rescan == RESCAN:
    #     rescan_fname = op.join(MMVT_DIR, mmvt_subject, 'fmri', 'fmri_CBF_rescan_rh.npy')
    #     target_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_CBF_rescan_rh.npy')
    #     utils.delete_file(target_fname)
    #     print('Copy {} to {}'.format(rescan_fname, target_fname))
    #     utils.copy_file(rescan_fname, target_fname)


@utils.tryit()
def morph_to_fsaverage(subject, scan_rescan, overwrite=False, print_only=False):
    for hemi in utils.HEMIS:
        source_fname = op.join(FMRI_DIR, subject, 'CBF_{}_{}.mgz'.format(scan_rescan, hemi))
        target_fname = op.join(FMRI_DIR, subject, 'CBF_{}_{}_fsaverage.mgz'.format(scan_rescan, hemi))
        if not op.isfile(target_fname) or overwrite:
            fu.surf2surf(subject, 'fsaverage', hemi, source_fname, target_fname, print_only=print_only)


def calc_cortical_histograms(subject, scan_rescan, atlas, low_threshold=40, high_threshold=100, overwrite=False,
                             do_plot=True, n_jobs=4):
    if not lu.check_labels(subject, atlas, SUBJECTS_DIR, MMVT_DIR):
        return False
    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'labels_hists_{}'.format(atlas)))
    output_fname = op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'labels_hists_{}.pkl'.format(atlas))
    if overwrite:
        utils.delete_folder_files(output_fol)
    print('Saving figures into {}'.format(output_fol))
    labels_data = {}
    for hemi in utils.HEMIS:
        npy_data_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_CBF_{}_{}.npy'.format(scan_rescan, hemi))
        if not op.isfile(npy_data_fname):
            print('Cannot find hemi data file! ({})'.format(npy_data_fname))
            return False
        surf_fname = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi))
        if not op.isfile(surf_fname):
            print('Cannot find surface file! ({})'.format(surf_fname))
            return False
        hemi_vals = np.load(npy_data_fname)
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi, n_jobs=n_jobs)
        labels_vertives_num = max([max(l.vertices) for l in labels])
        hemi_data_vertices_num = len(hemi_vals) - 1
        if labels_vertives_num != hemi_data_vertices_num:
            print('Wrong number of vertices or labels! labels_vertives_num ({}) != len({})'.format(
                labels_vertives_num, len(hemi_vals)))
            return False
        for label in tqdm(labels):
            fig_fname = op.join(output_fol, '{}.jpg'.format(label.name))
            if op.isfile(fig_fname) and not overwrite:
                continue
            labels_data[label.name] = label_data = hemi_vals[label.vertices]
            outliers = np.where(label_data < low_threshold)[0]
            if len(outliers) > 0:
                print('{} has {}/{} values < {}'.format(label.name, len(outliers), len(label_data), low_threshold))
            outliers = np.where(label_data > high_threshold)[0]
            if len(outliers) > 0:
                print('{} has {}/{} values > {}'.format(label.name, len(outliers), len(label_data), high_threshold))
            label_data[label_data < low_threshold] = low_threshold
            label_data[label_data > high_threshold] = high_threshold
            if do_plot:
                plt.hist(label_data, bins=40)
                plt.savefig(fig_fname)
                plt.close()
    utils.save(labels_data, output_fname)


def calc_scan_rescan_diff(
        subject, month, cortex_frac_threshold, do_plot=True, overwrite=False, check_if_should_run=False,
        verbose=True):
    input_fnames = [op.join(
        RESULTS_FOL, subject, '{}_gw_frac_{}_regions_stats.pkl'.format(
            get_full_subject_name(subject, month, scan_rescan), cortex_frac_threshold))
        for scan_rescan in SCAN_RESCAN]
    means_diff_output_fname = op.join(
        RESULTS_FOL, subject, '{}_gw_frac_{}_scan_rescan_diffs.pkl'.format(
            get_full_subject_name(subject, month), cortex_frac_threshold))
    mmvt_root_fol = utils.make_dir(op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data', 'subjects'))
    mmvt_diff_output_fname = op.join(mmvt_root_fol, '{}_gw_frac_{}_scan_rescan_diffs.npz'.format(
        get_full_subject_name(subject, month), cortex_frac_threshold))
    mmvt_rel_diff_output_fname = op.join(mmvt_root_fol, '{}_gw_frac_{}_scan_rescan_rel_diffs.npz'.format(
        get_full_subject_name(subject, month), cortex_frac_threshold))
    output_fnames = [means_diff_output_fname, mmvt_diff_output_fname, mmvt_rel_diff_output_fname]
    figure_output_fname = op.join(RESULTS_FOL, subject, 'month_{}_regions_scan_rescan_diffs.jpg'.format(month))
    do_run, ret = cics_checks(
        input_fnames, output_fnames, get_full_subject_name(subject, month), check_if_should_run, overwrite, verbose)
    if not do_run:
        return ret

    __, scan_means, _ = utils.load(input_fnames[0])
    __, rescan_means, _ = utils.load(input_fnames[1])
    region_names = scan_means.keys()
    diffs = {region: (scan_means[region] - rescan_means[region]
        if region in scan_means and region in rescan_means else 0) for region in region_names}
    if all([np.isnan(x) for x in diffs.values()]):
        return False
    rel_diffs = {region: ((scan_means[region] - rescan_means[region]) /
                          ((scan_means[region] + rescan_means[region]) / 2)
        if region in scan_means and region in rescan_means else 0) for region in region_names}
    regions_diffs = np.array([diffs[region_name] for region_name in region_names])
    rel_regions_diffs = np.array([rel_diffs[region_name] for region_name in region_names])
    labels_names = [get_aparc_label_name(region) for region in region_names]
    for data, output_fname in zip([regions_diffs, rel_regions_diffs],
                                  [mmvt_diff_output_fname, mmvt_rel_diff_output_fname]):
        title = utils.namebase(output_fname).replace('_', ' ')
        minmax = utils.calc_abs_minmax(data)
        np.savez(output_fname, names=labels_names, atlas='aparc',
                 data=data, title=title, data_min=-minmax, data_max=minmax, cmap='BuPu-YlOrRd')
    utils.save((diffs, rel_diffs), means_diff_output_fname)

    if do_plot and (not op.isfile(figure_output_fname) or overwrite):
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        x = range(len(labels_names))
        plt.bar(x, data)
        # plt.xticks(x, labels_names, rotation=90)
        plt.title('{} scan-rescan ASL diff'.format(subject))
        plt.ylabel('ASL diff')
        print('Saving bar plot in {}'.format(figure_output_fname))
        plt.savefig(figure_output_fname)
        plt.close()

    return all_files_exits(output_fnames)


def calc_scan_rescan_mean_diffs(
        subjects, site, months, cortex_frac_threshold, do_plot_hist, overwrite,
        check_if_should_run=False, verbose=False):
    # input
    input_fnames, input_files_num = {}, 0
    all_diffs, all_rel_diffs = defaultdict(list), defaultdict(list)
    for subject in subjects:
        for month in months:
            input_fname = op.join(
                RESULTS_FOL, subject, '{}_gw_frac_{}_scan_rescan_diffs.pkl'.format(
                    get_full_subject_name(subject, month), cortex_frac_threshold))
            if op.isfile(input_fname):
                try:
                    diffs, rel_diffs = utils.load(input_fname)
                    for region_name in diffs.keys():
                        all_diffs[region_name].append(diffs[region_name])
                        all_rel_diffs[region_name].append(rel_diffs[region_name])
                    input_fnames[subject] = input_fname
                except:
                    print('Error loading {} {} input file'.format(site, subject))
        print('{}-{}: {}/{} input files exist'.format(site, month, len(input_fnames), len(subjects)))
    if len(input_fnames) == 0:
        return False

    # output
    output_fnames = {}
    output_fol = utils.make_dir(op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data'))
    opers = ['mean', 'abs_med', 'std', 'rel-med', 'abs-rel-med']
    for oper in opers:
        output_fnames[oper] = op.join(
            output_fol, '{}_ASL_scan_rescan_diffs_{}_{}.npz'.format(site, oper, cortex_frac_threshold))

    cics_checks([], output_fnames, subject, check_if_should_run, overwrite, verbose)
    if check_if_should_run:
        return True

    region_names = all_diffs.keys()
    results_num = sum(~np.isnan(x) for x in all_diffs[list(all_diffs.keys())[0]])
    data_med = np.array([np.nanmedian(all_diffs[region_name]) for region_name in region_names])
    data_abs_med = np.array([np.abs(np.nanmedian(all_diffs[region_name])) for region_name in region_names])
    data_rel_med = np.array([np.nanmedian(all_rel_diffs[region_name]) for region_name in region_names])
    data_abs_rel_med = np.array([np.abs(np.nanmedian(all_rel_diffs[region_name])) for region_name in region_names])
    data_std = np.array([np.abs(all_diffs[region_name]).std() for region_name in region_names])
    labels_names = [get_aparc_label_name(region) for region in region_names]
    for data, oper in zip([data_med, data_abs_med, data_std, data_rel_med, data_abs_rel_med], opers):
        is_abs = oper in ['abs_med', 'abs-rel-med']
        minmax = utils.calc_abs_minmax(data)
        data_min = 0 if is_abs else -minmax
        print('Saving {}'.format(output_fnames[oper]))
        np.savez(output_fnames[oper], names=labels_names, atlas='aparc', data=data,
                 data_min=data_min, data_max=minmax, results_num=results_num,
                 cmap='YlOrRd' if is_abs else 'BuPu-RdOrYl', title='ASL_scan_rescan_diffs_{}'.format(oper))

        figure_output_fname = op.join(RESULTS_FOL, 'scan_rescan_stats', 'regions_scan_rescan_diffs_{}.jpg'.format(oper))
        if do_plot_hist and (not op.isfile(figure_output_fname) or overwrite):
            fig, ax = plt.subplots()
            x_pos = range(len(labels_names))
            ax.bar(x_pos, data, align='center', alpha=0.5, ecolor='black', capsize=10) # yerr=data_std / len(labels_names)
            plt.title('scan-rescan ASL {} diffs'.format(oper))
            plt.ylabel('ASL {} diff'.format(oper))
            plt.tight_layout()
            print('Saving bar plot in {}'.format(figure_output_fname))
            plt.savefig(figure_output_fname)
            plt.close()

    return all_files_exits(output_fnames)


def find_diff_clusters(subject, atlas='laus125', overwrite=True):
    clusters_name = 'CBF_scan_rescan'
    if overwrite:
        utils.delete_folder_files(
            op.join(MMVT_DIR, subject, 'fmri', 'clusters_labels_{}_{}'.format(clusters_name, atlas)),
            delete_folder=True)
        utils.delete_file(op.join(MMVT_DIR, subject, 'fmri', 'clusters_labels_{}_{}.pkl'.format(clusters_name, atlas)))
    fMRI.find_clusters(
        subject, 'CBF_scan_rescan', 2, 'laus125', 2, 1, create_clusters_labels=True,
        new_atlas_name='CBF_scan_rescan')


def remove_outliers(subject, scan_rescan):
    for hemi in utils.HEMIS:
        org_values_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_CBF_{}_{}.npy'.format(scan_rescan, hemi))
        z_values_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_CBF_scan_rescan_{}.npy'.format(hemi))
        if not op.isfile(org_values_fname) or not op.isfile(z_values_fname):
            print('remove_outliers: missing files!')
            continue
        org_values = np.load(org_values_fname)
        zvalues = np.load(z_values_fname)
        outliers_indices = np.where(zvalues > 2) if scan_rescan == SCAN else np.where(zvalues < -2)
        org_values[outliers_indices]


def detect_outliners(subject):
    # https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
    pass


def doubleMADsfromMedian(y,thresh=3.5):
    # http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh


def read_subject_hippocampus_volumes(subject):
    Left_Hippocampus_Code,  Right_Hippocampus_Code = 11, 26
    volumes = {'rh': [], 'lh': []}
    for base_6_12 in ['0', '6']: #, '12']:
        scan_rescan_vals = {}
        for scan_rescan in [SCAN, RESCAN]:
            scan_rescan_vals[scan_rescan] = {}
            aseg_fname = op.join(
                FS_BASE_6_ROOT, get_subject_fs_folder(subject, scan_rescan, base_6_12), 'stats', 'aseg.stats')
            if not op.isfile(aseg_fname):
                return None
            stat = np.loadtxt(aseg_fname, dtype="i1,i1,i4,f4,S32,f4,f4,f4,f4,f4")
            assert(stat[Left_Hippocampus_Code][4].decode() == 'Left-Hippocampus')
            assert (stat[Right_Hippocampus_Code][4].decode() == 'Right-Hippocampus')
            scan_rescan_vals[scan_rescan]['lh'] = stat[Left_Hippocampus_Code][5]
            scan_rescan_vals[scan_rescan]['rh'] = stat[Right_Hippocampus_Code][5]
        for hemi in utils.HEMIS:
            # volumes[hemi].append((scan_rescan_vals[SCAN][hemi] + scan_rescan_vals[RESCAN][hemi]) / 2.0)
            if base_6_12 == '0':
                volumes[hemi].append(min(scan_rescan_vals[SCAN][hemi], scan_rescan_vals[RESCAN][hemi]))
            elif base_6_12 == '6':
                volumes[hemi].append(max(scan_rescan_vals[SCAN][hemi], scan_rescan_vals[RESCAN][hemi]))
    return volumes


def plot_registration_cost_hist(subjects, site, overwrite=False):
    fig_fname = op.join(RESULTS_FOL, site, 'reg_costs.jpg')
    csv_fname = op.join(RESULTS_FOL, site, 'reg_costs.csv')
    results, mincosts = [], []
    for subject in subjects:
        for scan_rescan in [SCAN, RESCAN]:
            mincost_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'control_to_T1.dat.mincost')
            if not op.isfile(mincost_fname):
                print('{} does not exist!'.format(mincost_fname))
                continue
            score = np.genfromtxt(mincost_fname)[0]
            results.append([score, subject, scan_rescan])
            mincosts.append(score)
    plt.hist(mincosts)
    plt.savefig(fig_fname)
    plt.close()

    import csv
    print('writing {}'.format(csv_fname))
    results.sort(key=lambda res: res[0])
    with open(csv_fname, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for res in results[::-1]:
            csv_writer.writerow(res)


def calc_volume_fractions(
        subject, subject_fol, scan_rescan, month, overwrite=False, print_only=False, check_if_should_run=False, verbose=True):
    reg_fname = op.join(subject_fol, 'control_to_T1.lta')
    output_fname = op.join(subject_fol, 'CBF')
    output_mgz_fname = op.join(subject_fol, 'CBF.cortex.mgz')
    do_run, ret = cics_checks(
        [reg_fname], [output_mgz_fname], subject_fol, check_if_should_run, overwrite, verbose)
    if not do_run:
        return ret
    set_subject(subject, scan_rescan, month)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # mri_compute_volume_fractions_reg = 'mri_compute_volume_fractions --o "{output_fname}" --reg  "{reg_fname}"'
    rs(mri_compute_volume_fractions_reg)
    return all_files_exits([output_mgz_fname])


def plot_subjects_cbf_histograms(
        subjects, site, bandwidth=0.2, overwrite=False, per_subject=False, do_plot=True,
        check_if_should_run=False, verbose=False):
    # input
    input_fnames, input_files_num = {}, 0
    for subject in subjects:
        for scan_rescan in [SCAN, RESCAN]:
            input_files_num += 1
            input_fname = op.join(RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan, 'aparc_results.pkl')
            if op.isfile(input_fname):
                input_fnames[(subject, scan_rescan)] = input_fname
    print('{}: {}/{} input files exist'.format(site, len(input_fnames), input_files_num))
    if len(input_fnames) == 0:
        return False
    # output
    plots_output_fol = utils.make_dir(op.join(RESULTS_FOL, site, 'hists', 'cbf_aparc_hists{}'.format(
        '_per_subject' if per_subject else '')))
    stat_output_fol = utils.make_dir(op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data'))
    cbf_aparc_hists_output_fname = op.join(RESULTS_FOL, site, 'cbf_aparc_hists.pkl')
    stat_output_fnames = [
        op.join(stat_output_fol, '{}_{}.npz'.format(site, file_name)) for file_name in
        ['CBF_mean', 'CBF_hist_var', 'CBF_outliers', 'CBF_shapiro', 'CBF_cortical_frac_mean']]
    output_fnames = [cbf_aparc_hists_output_fname, *stat_output_fnames]
    cics_checks([], output_fnames, subject, check_if_should_run, overwrite, verbose)
    if check_if_should_run:
        return True

    x_grid = np.linspace(-50, 150, 200)
    all_regions_values, all_regions_fracs = defaultdict(list), defaultdict(list)
    output_str = ''
    if not op.isfile(cbf_aparc_hists_output_fname) or overwrite:
        kdes, all_kdes = defaultdict(list), {}
        for subject in tqdm(subjects):
            for scan_rescan in [SCAN, RESCAN]:
                input_fname = input_fnames.get((subject, scan_rescan), '')
                if not op.isfile(input_fname):
                    output_str += 'No hist file for {} {}\n'.format(subject, scan_rescan)
                    continue
                regions_values, regions_means, regions_fracs = utils.load(input_fname)
                # regions_values =
                for region in regions_values.keys():
                    all_regions_values[region].extend(regions_values[region])
                    all_regions_fracs[region].append(regions_fracs[region])
                    if per_subject:
                        if len(regions_values[region]) > 1:
                            kde = utils.kde(regions_values[region], x_grid, bandwidth=bandwidth)
                            kdes[region].append(kde)
                        else:
                            output_str += '{} {} has no values!\n'.format(subject, region)
        for region in tqdm(regions_values.keys()):
            if per_subject:
                kdes[region] = np.array(kdes[region])
            all_regions_values[region] = np.array(all_regions_values[region])
            all_regions_fracs[region] = np.array(all_regions_fracs[region])
            if len(all_regions_values[region]) > 1:
                all_kdes[region] = utils.kde(all_regions_values[region], x_grid, bandwidth=bandwidth).squeeze()
            else:
                output_str += '{}: {} has no values!\n'.format(site, region)
        utils.save((kdes, all_kdes, all_regions_values, all_regions_fracs), cbf_aparc_hists_output_fname)
    else:
        kdes, all_kdes, all_regions_values, all_regions_fracs = utils.load(cbf_aparc_hists_output_fname)
    if verbose:
        print(output_str)

    if do_plot:
        if per_subject:
            for region_name, region_kdes in tqdm(kdes.items()):
                figure_fname = op.join(plots_output_fol, '{}.jpg'.format(region_name))
                if op.isfile(figure_fname) and not overwrite:
                    continue
                plt.figure()
                for region_kde in region_kdes:
                    plt.plot(x_grid, region_kde)
                plt.savefig(figure_fname)
                plt.close()
        else:
            for region_name, region_kde in tqdm(all_kdes.items()):
                figure_fname = op.join(plots_output_fol, '{}.jpg'.format(region_name))
                if op.isfile(figure_fname) and not overwrite:
                    continue
                plt.figure()
                plt.plot(x_grid, region_kde)
                plt.savefig(figure_fname)
                plt.close()
        print('Figures were saved in {}'.format(plots_output_fol))

    from scipy.stats import shapiro
    data_mean, data_var, outliers, region_names, shapiro_p_vals, cotical_fracs = [], [], [], [], [], []
    for region in all_regions_values.keys():
        if len(all_regions_values[region]) < 5:
            print('{}: only {} values for {}!!'.format(site, len(all_regions_values[region]), region))
            continue
        region_values = np.array(all_regions_values[region])
        region_names.append(get_aparc_label_name(region))
        data_mean.append(region_values.mean())
        data_var.append(region_values.var())
        outliers.append((len(np.where(region_values > high_threshold)[0]) * 100) / len(region_values))
        shapiro_p_vals.append(shapiro(region_values)[1])
        cotical_fracs.append(np.array(all_regions_fracs[region]).mean())
    for data, output_fname in zip([data_mean, data_var, outliers, shapiro_p_vals, cotical_fracs], stat_output_fnames):
        np.savez(output_fname, names=region_names, atlas='aparc',
                 data=data, title=utils.namebase(output_fname).replace('_', ' '),
                 data_min=np.min(data), data_max=np.max(data), cmap='YlOrRd')
    return all([op.isfile(fname) for fname in output_fnames])
    '''
    if shapiro_p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    '''


def get_aparc_label_name(region_name):
    return '-'.join(region_name.split('-')[1:][::-1])


@utils.tryit(except_retval=[])
def get_subjects(site):
    # subjects = set([fol.split('_')[0].replace('-base', '') for fol in utils.get_subfolders(FS_ROOT, 'name')])
    # return sorted(list(subjects - set(['scripts', 'fsaverage'])))
    subjects = [fol.split('_')[0] for fol in utils.get_subfolders(op.join(HOME_FOL, site), 'name')]
    return subjects


def read_hippocampus_volumes(overwrite=False):
    output_fname = op.join(HOME_FOL, 'hippocampus_volumes.pkl')
    if not op.isfile(output_fname) or overwrite:
        subjects = get_subjects()
        all_volumes = {'rh': [], 'lh': []}
        for subject in tqdm(subjects):
            volumes = read_subject_hippocampus_volumes(subject)
            if volumes is None:
                continue
            for hemi in utils.HEMIS:
                all_volumes[hemi].append(volumes[hemi])
        utils.save(all_volumes, output_fname)
    else:
        all_volumes = utils.load(output_fname)
    for hemi in utils.HEMIS:
        x = np.array(all_volumes[hemi])
        plt.hist(np.diff(x))
        plt.title(hemi)
        plt.show()


def vars_names_to_dict(params, args):
    return {key: args[key] for key in params}


def run_functions_in_parallel(funcs, all_params, n_jobs, ask_to_continue=True, split_jobs=True, verbose=True,
                              overwrite=False):
    for func in funcs:
        run_function_in_parallel(func, all_params, n_jobs, ask_to_continue, split_jobs, verbose, overwrite)


def run_function_in_parallel(func, all_params, n_jobs, ask_to_continue=True, split_jobs=True, verbose=True,
                             overwrite=False):
    for params in all_params:
        params['overwrite'] = overwrite
    # Filter our items that should not run (all input files should exist, and all output files shoud not)
    params = [p for p in all_params if all(_run_func_in_parallel((func, [p], True, 0, verbose)))]
    print('*** Run {}/{} records, {} jobs ***'.format(len(params), len(all_params), n_jobs))
    if ask_to_continue:
        ret = input('ok? (y/n) ')
        if not au.is_true(ret):
            return
    if split_jobs:
        chunks_indices = np.array_split(np.arange(len(params)), n_jobs)
        chunk_params = [(func, [params[ind] for ind in chunk_indices], False, thread_ind, verbose)
                        for thread_ind, chunk_indices in enumerate(chunks_indices)]
    else:
        chunk_params = [(func, [p], False, thread_ind, verbose) for thread_ind, p in enumerate(params)]
    results = utils.run_parallel(_run_func_in_parallel, chunk_params, n_jobs)
    results = utils.combine_chunks(results)
    print('{}/{} good results'.format(sum(results), len(params)))


def _run_func_in_parallel(parallel_params):
    func, params, check_if_should_run, thread_ind, verbose = parallel_params
    flags = []
    now = time.time()
    for run_num, p in enumerate(params):
        if not check_if_should_run:
            utils.time_to_go(now, run_num, len(params), 1, thread_ind)
        # Check if the params are per subject
        flag = False
        if 'subject' in p:
            subject, site, month, cortex_frac_threshold = [
                p[key] for key in ['subject', 'site', 'month', 'cortex_frac_threshold']]
            scan_rescan = p.get('scan_rescan', SCAN)
            subject_dir = p.get('subject_dir', '')
            low_threshold, high_threshold = [p.get(key, 0) for key in ['low_threshold', 'high_threshold']]
            overwrite, print_only, do_plot = [p.get(key, False) for key in ['overwrite', 'print_only', 'do_plot']]
            if not op.isdir(subject_dir):
                utils.make_dir(subject_dir)
            if func.__name__ == 'register_cbf_to_t1':
                flag = register_cbf_to_t1(
                    subject, subject_dir, scan_rescan,  month, overwrite, print_only, check_if_should_run, verbose)
            elif func.__name__ == 'calc_volume_fractions':
                flag = calc_volume_fractions(
                    subject, subject_dir, scan_rescan, month, overwrite, print_only, check_if_should_run, verbose)
            elif func.__name__ == 'register_aseg_to_cbf':
                flag = register_aseg_to_cbf(
                    subject, subject_dir, scan_rescan, month, overwrite, print_only, check_if_should_run, verbose)
            elif func.__name__ == 'calc_regions_stats':
                flag = calc_regions_stats(
                    subject, subject_dir, scan_rescan, month, low_threshold, high_threshold, cortex_frac_threshold,
                    overwrite, do_plot, check_if_should_run, verbose)
            elif func.__name__ == 'calc_scan_rescan_diff':
                flag = calc_scan_rescan_diff(
                    subject, month, cortex_frac_threshold, do_plot, overwrite, check_if_should_run, verbose)
            elif func.__name__ == 'calc_T1_CNR':
                flag = calc_T1_CNR(
                    subject, subject_dir, scan_rescan, month, print_only, overwrite, check_if_should_run, verbose)

        # Check of the parameters are per site
        elif 'subjects' in p and 'site' in p:
            subjects, site, month, months, cortex_frac_threshold = [p[key] for key in [
                'subjects', 'site', 'month', 'months', 'cortex_frac_threshold']]
            overwrite, print_only, do_plot, per_subject = [p.get(key, False) for key in [
                'overwrite', 'print_only', 'do_plot', 'per_subject']]
            if func.__name__ == 'plot_subjects_cbf_histograms':
                bandwidth = p.get('bandwidth', 0.2)
                flag = plot_subjects_cbf_histograms(
                    subjects, site, bandwidth, overwrite, per_subject, do_plot, check_if_should_run, verbose)
            if func.__name__ == 'calc_scan_rescan_mean_diffs':
                flag = calc_scan_rescan_mean_diffs(
                    subjects, site, months, cortex_frac_threshold, do_plot, overwrite, check_if_should_run, verbose)
            if func.__name__ == 'average_T1_CNR':
                flag = average_T1_CNR(subjects, site, month, do_plot, overwrite, check_if_should_run, verbose)
        flags.append(flag)
    return flags


def get_subject_months_dirs(subject, site, months):
    return [op.join(HOME_FOL, site, '{}{}'.format(subject, month)) for month in months
            if op.isdir(op.join(HOME_FOL, site, '{}{}'.format(subject, month)))]


def get_subject_dir(subject, site, month, scan_rescan):
    return op.join(HOME_FOL, site, '{}{}'.format(subject, '_{}'.format(month) if month != '0' else ''),
                   scan_rescan)


def calc_different_cortec_frac_thresholds(
        sites, months, cortex_frac_thresholds, n_jobs, low_threshold=0, high_threshold=100, overwrite=False):
    # for cortex_frac_threshold in cortex_frac_thresholds:
    for cortex_frac_threshold in [0.9]:
        params, scan_rescan_params, site_params = get_params(
            sites, months, cortex_frac_threshold, low_threshold, high_threshold, overwrite, n_jobs=n_jobs)
        # run_functions_in_parallel(
        #     [calc_regions_stats, calc_scan_rescan_diff], params, n_jobs, ask_to_continue=False, verbose=False)
        run_function_in_parallel(
            calc_scan_rescan_mean_diffs, site_params, n_jobs, overwrite=True, ask_to_continue=False, verbose=False,
            split_jobs=False)


    for site in sites:
        input_fol = op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data')
        opers = ['rel-med', 'abs-rel-med']
        for oper in opers:
            vals, results_nums = [], []
            for cortex_frac_threshold in cortex_frac_thresholds:
                input_fname = op.join(
                    input_fol, '{}_ASL_scan_rescan_diffs_{}_{}.npz'.format(site, oper, cortex_frac_threshold))
                d = np.load(input_fname)
                vals.append(float(np.nanmedian(d['data'])) * 100)
                results_nums.append(int(d['results_num']))
            title = '{}: ASL scan rescan diffs {}'.format(site, oper)
            sub_fol = 'abs_scan_rescan_diffs' if 'abs' in oper else 'scan_rescan_diffs'
            figures_fol = utils.make_dir(op.join(RESULTS_FOL, 'all_sites_results', sub_fol))
            figure_fname = op.join(figures_fol, '{}_{}_scan_rescans_diffs.jpg'.format(site, oper))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(cortex_frac_thresholds, vals)
            if 'abs' in oper:
                ax1.set_ylim(bottom=0)
            ax1.set_ylabel('rel scan-rescan diff (%)')
            # ax2 = ax1.twinx()
            # ax2.scatter(cortex_frac_thresholds, results_nums, marker='+', c='yellow')
            # # ax2.set_ylim(bottom=min(results_nums) - 1)
            # ax2.set_ylabel('#results')
            plt.title(title)
            print('Saving {}'.format(figure_fname))
            plt.savefig(figure_fname)
            plt.close()


def get_params(sites, months, cortex_frac_threshold=0.9, low_threshold=0, high_threshold=100,
               overwrite=False, print_only=False, do_plot=False, run_preproc_anat=False, n_jobs=4):
    params, scan_rescan_params, site_params = [], [], []
    for site in sites:
        for month in months:
            subjects = get_subjects(site)
            site_params.append(vars_names_to_dict((
                'subjects', 'site', 'month', 'months', 'cortex_frac_threshold', 'overwrite', 'do_plot'), locals()))
            for sub_ind, subject in enumerate(subjects):
                scan_rescan_params.append(vars_names_to_dict((
                    'subject', 'site', 'month', 'cortex_frac_threshold', 'overwrite', 'do_plot'), locals()))
                for scan_rescan in SCAN_RESCAN:
                    if run_preproc_anat:
                        ret = preproc_anat(subject, month, scan_rescan, overwrite_files=False)
                        if not ret:
                            continue
                    subject_dir = get_subject_dir(subject, site, month, scan_rescan)
                    params.append(vars_names_to_dict((
                        'subject', 'site', 'subject_dir', 'scan_rescan', 'month', 'low_threshold', 'high_threshold',
                        'cortex_frac_threshold', 'overwrite', 'print_only', 'do_plot'), locals()))
    return params, scan_rescan_params, site_params


def main(sites, months, n_jobs=4):
    params, scan_rescan_params, site_params = get_params(
        sites, months, cortex_frac_threshold=0.9, low_threshold=0, high_threshold=100,
        overwrite=False, print_only=False, do_plot=False, run_preproc_anat=False, n_jobs=n_jobs)
    funcs = [register_cbf_to_t1, calc_volume_fractions, register_aseg_to_cbf, calc_regions_stats]
    run_functions_in_parallel(funcs, params, n_jobs, ask_to_continue=True, split_jobs=True, verbose=True)
    run_function_in_parallel(calc_scan_rescan_diff, scan_rescan_params, n_jobs)
    run_function_in_parallel(calc_scan_rescan_mean_diffs, site_params, n_jobs, split_jobs=False)


if __name__ == '__main__':
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
    sites = ['131-NeuroBeh_ACH', '277-NDC', '800-Hoglund', '829-EmoryUniversity', '960-VitalImaging']
    months = ['0', '6', '12']
    cortex_frac_thresholds = np.linspace(0.5, 0.9, 17)
    n_jobs = utils.get_n_jobs(30)
    print('n_jobs: {}'.format(n_jobs))

    # main(sites, months, cortex_frac_threshold=0.9, overwrite=True, n_jobs=n_jobs)
    # calc_different_cortec_frac_thresholds(sites, months, cortex_frac_thresholds, n_jobs)

    params, scan_rescan_params, site_params = get_params(
        sites, months, cortex_frac_threshold=0.5, low_threshold=0, high_threshold=100,
        print_only=False, do_plot=False, run_preproc_anat=False, n_jobs=n_jobs)
    # run_function_in_parallel(calc_T1_CNR, params, n_jobs, overwrite=False)
    run_function_in_parallel(average_T1_CNR, site_params, n_jobs, overwrite=False)

    # plot_global_data(sites, months, cortex_frac_threshold=0.5, overwrite=True)

    # read_hippocampus_volumes()
    # calc_volume_fractions_all_subjects(subjects, site, overwrite, print_only)
    # plot_cortex_frac_vs_diff(sites)







