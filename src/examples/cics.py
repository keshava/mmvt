import os.path as op
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
from collections import defaultdict

from src.utils import utils
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu
from src.utils import labels_utils as lu
from src.preproc import fMRI
from src.preproc import anatomy as anat

FS_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_recons'
FS_BASE_6_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_6month_recons'
HOME_FOL = '/autofs/space/nihilus_001/CICS/users/noam/CICS/'
RESULTS_FOL = [d for d in ['/autofs/space/nihilus_001/CICS/users/noam/figures/', 'C:\\Users\\peled\\CICS']
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


def get_subject_fs_folder(subject, scan_rescan, base_6_12='0'):
    base_6_12_str = '' if base_6_12 == '0' else base_6_12
    scan_rescan_str = base_6_12_str if scan_rescan == SCAN else '{}B'.format(base_6_12_str)
    if scan_rescan_str != '':
        scan_rescan_str += '_'
    return '{0}_{1}recon.long.{0}-base'.format(subject, scan_rescan_str)


def register_cbf_to_t1(subject, subject_fol, scan_rescan, overwrite=False, print_only=False, check_if_should_run=False,
                       verbose=True):

    def input_files_exist():
        return op.isfile(source_fname) and op.isfile(target_fname)

    def output_exists():
        if op.isfile(lta_fname) and op.isfile(output_fname):
            try:
                nib.load(output_fname)
                nib.load(lta_fname)
            except:
                return False
        return not overwrite

    subject_fol = op.join(subject_fol, scan_rescan)
    output_fname = op.join(subject_fol, 'Control_to_T1.nii')
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    source_fname = op.join(subject_fol, 'Control.nii')
    target_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.mgz')
    if overwrite:
        utils.delete_file(output_fname)
        utils.delete_file(lta_fname)

    if check_if_should_run:
        if verbose and not input_files_exist():
            print('register_cbf_to_t1: input files does not exist! ({})'.format(subject_fol))
        if verbose and output_exists():
            print('register_cbf_to_t1: output exist, continue ({})'.format(subject_fol))
        return input_files_exist() and not output_exists()

    tmp_fol = utils.make_dir(op.join(subject_fol, 'bbregister'))
    if not input_files_exist():
        print('The source ({}) does not exist'.format(source_fname))
        print('And/Or the target ({}) does not exist!'.format(target_fname))
        return False
    if output_exists():
        return True
    # Set the subject for the bbregister
    subject = subject if scan_rescan == SCAN else '{}_rescan'.format(subject)
    os.environ['SUBJECT'] = subject
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # bbregister --s {subject} --mov "{source_fname}" --lta "{lta_fname}" --t1 --o "{output_fname}"
    rs(bbregister)

    if not op.isfile(lta_fname):
        print('No registration file found! ({})'.format(lta_fname))
        return False
    else:
        return True


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


def register_aseg_to_cbf(subject, site, scan_rescan, overwrite=False, print_only=True):
    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    source_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'aparc+aseg.mgz')
    output_fname = op.join(subject_fol, 'aparc+aseg_cbf.mgz')
    if not op.isfile(output_fname) or overwrite:
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(register_using_inverse_lta)
    else:
        print('{} already exist'.format(output_fname))


def calc_cbf_histograms(subject, scan_rescan, low_threshold, high_threshold, cortex_frac_threshold=0.9,
                        overwrite=False, do_plot=True):
    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    output_fol = utils.make_dir(op.join(op.join(RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan)))
    values_output_fname = op.join(op.join(
        RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan, 'aparc_aseg_hist.pkl'))
    means_output_fname = op.join(op.join(
        RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan, 'aparc_values.pkl'))
    if op.isfile(values_output_fname) and op.isfile(means_output_fname) and not overwrite and not do_plot:
        return True
    cics_cbf_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF.nii')
    if not op.isfile(cics_cbf_fname):
        print('calc_cbf_histograms: Cannot find {}!'.format(cics_cbf_fname))
        return False
    aparc_aseg_fname = op.join(subject_fol, 'aparc+aseg_cbf.mgz')
    if not op.isfile(aparc_aseg_fname):
        print('calc_cbf_histograms: Cannot find {}!'.format(aparc_aseg_fname))
        return False
    cortex_frac_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF.cortex.mgz')
    if not op.isfile(cortex_frac_fname):
        print('No cortex frac!')
        return False
    cbf_data = nib.load(cics_cbf_fname).get_data()
    cortex_frac = nib.load(cortex_frac_fname).get_data()
    lut = utils.read_freesurfer_lookup_table(return_dict=True)
    aparc_aseg = nib.load(aparc_aseg_fname).get_data()
    unique_codes = list(range(1001, 1036)) + list(range(2001, 2036))
    if overwrite and do_plot:
        utils.delete_folder_files(output_fol)
    regions_values, regions_means = {}, {}
    output_str = ''
    for code in tqdm(unique_codes):
        region_name = lut.get(code, None)
        if region_name is None:
            # print('{} not in lut!'.format(code))
            continue
        fig_fname = op.join(output_fol, '{}.jpg'.format(region_name))
        if op.isfile(fig_fname) and not overwrite:
            continue
        region_values = cbf_data[np.where(aparc_aseg == code)]
        cortex_frac_values = cortex_frac[np.where(aparc_aseg == code)]
        cortex_indices = np.where(cortex_frac_values > cortex_frac_threshold)
        if len(cortex_indices[0]) == 0:
            output_str += '{} has no voxels with cortex_frac > {}!\n'.format(region_name, cortex_frac_threshold)
            continue
        region_values = region_values[cortex_indices]
        # regions_values[region_name] = region_values.copy()
        in_bounderies_indices = np.where((low_threshold < region_values) & (region_values < high_threshold))
        region_values = region_values[in_bounderies_indices]
        regions_values[region_name] = region_values.copy()
        regions_means[region_name] = np.mean(regions_values[region_name])
        # if len(outliers) > 0:
        #     output_str += '{} has {} out of {} values < {}\n'.format(
        #         region_name, len(outliers), len(region_values), low_threshold)
        # outliers = np.where(region_values > high_threshold)[0]
        # if len(outliers) > 0:
        #     output_str += '{} has {} out of {} values > {}\n'.format(
        #         region_name, len(outliers), len(region_values), high_threshold)
        # region_values[region_values < low_threshold] = low_threshold
        # region_values[region_values > high_threshold] = high_threshold
        if do_plot:
            plt.hist(region_values, bins=40)
            plt.savefig(fig_fname)
            plt.close()
    print(output_str)
    utils.save(regions_values, values_output_fname)
    utils.save(regions_means, means_output_fname)


def _plot_cbf_histograms_parallel(p):
    subject, scan_rescan = p
    input_fname = op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'aparc_aseg_hist.pkl')
    output_fol = utils.make_dir(op.join(RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan))
    if not op.isfile(input_fname):
        return False
    regions_values = utils.load(input_fname)
    for region_name, region_values in regions_values.values():
        fig_fname = op.join(output_fol, '{}.jpg'.format(region_name))
        plt.hist(region_values, bins=40)
        plt.savefig(fig_fname)
        plt.close()


# def print_freeview_cmd(subject, subject_fol):
#     cbf = op.join(subject_fol, 'CBF_to_T1.nii')
#     control = op.join(subject_fol, 'Control_to_T1.nii')
#     t1 = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
#     print('freeview {} {} {}'.format(cbf, control, t1))

def get_labels_data(subject, atlas):
    necessary_files = {'label': ['{}.{}.annot'.format(hemi, atlas) for hemi in utils.HEMIS]}
    remote_subject_dir = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject))
    utils.prepare_subject_folder(necessary_files, subject, remote_subject_dir, SUBJECTS_DIR)


def preproc_anat(subjects, overwrite_files=False):
    good_subjects = []
    for subject in subjects:
        # utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
        # utils.make_dir(op.join(MMVT_DIR, '{}_rescan'.format(subject), 'fmri'))
        scan_ret = utils.prepare_subject_folder(
            anat.get_necessary_files(), subject,
            op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject)), SUBJECTS_DIR,
            overwrite_files=overwrite_files)[0]
        rescan_ret = utils.prepare_subject_folder(
            anat.get_necessary_files(), '{}_rescan'.format(subject),
            op.join(FS_ROOT, '{0}_B_recon.long.{0}-base'.format(subject)), SUBJECTS_DIR,
            overwrite_files=overwrite_files)[0]
        if scan_ret and rescan_ret:
            good_subjects.append(subject)
    print('{}/{} good subjects'.format(len(good_subjects), len(subjects)))
    return good_subjects


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


def calc_scan_rescan_diff(subject, do_plot_hist=True, overwrite=False):
    means_input_fnames = [op.join(op.join(
        RESULTS_FOL, 'aparc_aseg_hists', subject, scan_rescan, 'aparc_values.pkl')) for scan_rescan in SCAN_RESCAN]
    if not all([op.isfile(fname) for fname in means_input_fnames]):
        print('calc_scan_rescan_diff: {} no all files exist!'.format(subject))
        return
    means_diff_fname = op.join(
        RESULTS_FOL, 'aparc_aseg_hists', subject, 'aparc_values_diffs.pkl')
    mmvt_file_name = '{}_ASL_scan_rescan_diffs'.format(subject)
    mmvt_output_fname = op.join(utils.make_dir(
        op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data')),
        '{}.npz'.format(mmvt_file_name))
    if op.isfile(mmvt_output_fname) and op.isfile(means_diff_fname) and not overwrite:
        print('calc_scan_rescan_diff: files exist for {}'.format(subject))
        return True
    scan_means, rescan_means = [utils.load(fname) for fname in means_input_fnames]
    region_names = scan_means.keys()
    diffs = {region: scan_means.get(region, 0) - rescan_means.get(region, 0) for region in region_names}
    utils.save(diffs, means_diff_fname)
    data = np.array([diffs[region_name] for region_name in region_names])
    labels_names = [get_aparc_label_name(region) for region in region_names]
    minmax = utils.calc_abs_minmax(data)

    figure_output_fname = op.join(RESULTS_FOL, 'aparc_aseg_hists', subject, 'labels_scan_rescan_diffs.jpg')
    if do_plot_hist and (not op.isfile(figure_output_fname) or overwrite):
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

    np.savez(mmvt_output_fname, names=labels_names, atlas='aparc',
             data=data, title=mmvt_file_name, data_min=-minmax, data_max=minmax, cmap='BuPu-YlOrRd')
    # fMRI.calc_files_diff(
    #     subject, 'fmri_CBF_scan_{hemi},fmri_CBF_rescan_{hemi}', 'CBF_scan_rescan', 'zvals', overwrite)


def calc_scan_rescan_mean_diffs(subjects, do_plot_hist, overwrite):
    all_diffs = defaultdict(list)
    for subject in subjects:
        means_diff_fname = op.join(
            RESULTS_FOL, 'aparc_aseg_hists', subject, 'aparc_values_diffs.pkl')
        if not op.isfile(means_diff_fname):
            print('No diffs file for {}!'.format(subject))
            continue
        diffs = utils.load(means_diff_fname)
        for region_name, val in diffs.items():
            all_diffs[region_name].append(val)
    region_names = all_diffs.keys()
    data_mean = np.array([np.abs(all_diffs[region_name]).mean() for region_name in region_names])
    data_std = np.array([np.abs(all_diffs[region_name]).std() for region_name in region_names])
    labels_names = [get_aparc_label_name(region) for region in region_names]

    figure_output_fname = op.join(RESULTS_FOL, 'aparc_aseg_hists', 'labels_scan_rescan_diffs.jpg')
    if do_plot_hist and (not op.isfile(figure_output_fname) or overwrite):
        fig, ax = plt.subplots()
        x_pos = range(len(labels_names))
        ax.bar(x_pos, data_mean, align='center', alpha=0.5, ecolor='black', capsize=10) # yerr=data_std / len(labels_names)
        plt.title('scan-rescan ASL mean diffs')
        plt.ylabel('ASL diff')
        plt.tight_layout()
        print('Saving bar plot in {}'.format(figure_output_fname))
        plt.savefig(figure_output_fname)
        plt.close()

    for data, oper in zip([data_mean, data_std], ['mean', 'std']):
        mmvt_file_name = 'ASL_scan_rescan_diffs_{}'.format(oper)
        mmvt_output_fname = op.join(utils.make_dir(
            op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data')), '{}.npz'.format(mmvt_file_name))
        if op.isfile(mmvt_output_fname) and not overwrite:
            continue
        np.savez(mmvt_output_fname, names=labels_names, atlas='aparc', data=data, title=mmvt_file_name,
                 data_min=0, data_max=data.max(), cmap='YlOrRd')


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


def _calc_volume_fractions_parallel(params):
    for p in params:
        subject, site, subject_dir, scan_rescan = [p[key] for key in ['subject', 'site', 'subject_dir', 'scan_rescan']]
        overwrite, print_only = [p.get(key, False) for key in ['overwrite', 'print_only']]
        calc_volume_fractions(subject, site, subject_dir, scan_rescan,  overwrite, print_only)


def calc_volume_fractions(subject, site, subject_dir, scan_rescan, overwrite=False, print_only=False):
    # 'mri_compute_volume_fractions --o "{output_fname}" --regheader {subject} "{target_fname}"'
    # mri_compute_volume_fractions_reg = 'mri_compute_volume_fractions --o "{output_fname}" --reg  "{reg_fname}"'
    # if use_reg:
    reg_fname = op.join(subject_dir, scan_rescan, 'control_to_T1.lta')
    output_fname = op.join(subject_dir, scan_rescan, 'CBF')
    rs = utils.partial_run_script(locals(), print_only=print_only)
    output_full_fname = op.join(subject_dir, scan_rescan, 'CBF.cortex.mgz')
    # else:
    #     output_fname = op.join(subject_dir, scan_rescan, 'T1')
    #     target_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.mgz')
    #     rs = utils.partial_run_script(locals(), print_only=print_only)
    #     output_full_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.cortex.mgz')
    if not op.isfile(output_full_fname) or overwrite:
        rs(mri_compute_volume_fractions_reg)



def plot_subjects_cbf_histograms(subjects, site, bandwidth=0.2, overwrite=False, per_subject=False):
    from collections import defaultdict

    output_fol = utils.make_dir(op.join(RESULTS_FOL, site, 'hists', 'cbf_aparc_hists{}'.format(
        '_per_subject' if per_subject else '')))
    output_fname = op.join(RESULTS_FOL, site, 'cbf_aparc_hists.pkl')
    x_grid = np.linspace(-50, 150, 200)
    stats = defaultdict(dict)
    all_regions_values = defaultdict(list)
    output_str = ''
    if not op.isfile(output_fname) or overwrite:
        kdes, all_kdes = defaultdict(list), {}
        for subject in tqdm(subjects):
            for scan_rescan in [SCAN, RESCAN]:
                input_fname = op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'aparc_aseg_hist.pkl')
                if not op.isfile(input_fname):
                    output_str += 'No hist file for {} {}\n'.format(subject, scan_rescan)
                    continue
                regions_values = utils.load(input_fname)
                for region, values in regions_values.items():
                    all_regions_values[region].extend(values)
                    if per_subject:
                        if len(values) > 1:
                            kde = utils.kde(values, x_grid, bandwidth=bandwidth)
                            kdes[region].append(kde)
                        else:
                            output_str += '{} {} has no values!\n'.format(subject, region)
        for region in regions_values.keys():
            kdes[region] = np.array(kdes[region])
            all_regions_values[region] = np.array(all_regions_values[region])
            all_kdes[region] = utils.kde(all_regions_values[region], x_grid, bandwidth=bandwidth).squeeze()

            # w = np.mean(x, axis=0)
            # stats[region]['mean'] = region_mean = np.average(x_grid, weights=w)
            # stats[region]['var'] = np.average((region_mean - x_grid) ** 2, weights=w)
        utils.save((kdes, all_kdes, all_regions_values), output_fname)
    else:
        kdes, all_kdes, all_regions_values = utils.load(output_fname)

    print(output_str)
    if per_subject:
        for region_name, region_kdes in tqdm(kdes.items()):
            figure_fname = op.join(output_fol, '{}.jpg'.format(region_name))
            if op.isfile(figure_fname) and not overwrite:
                continue
            plt.figure()
            for region_kde in region_kdes:
                plt.plot(x_grid, region_kde)
            plt.savefig(figure_fname)
            plt.close()
    else:
        for region_name, region_kde in tqdm(all_kdes.items()):
            figure_fname = op.join(output_fol, '{}.jpg'.format(region_name))
            if op.isfile(figure_fname) and not overwrite:
                continue
            plt.figure()
            plt.plot(x_grid, region_kde)
            plt.savefig(figure_fname)
            plt.close()

    print('Figures were saved in {}'.format(output_fol))

    from scipy.stats import shapiro
    fol = utils.make_dir(op.join(MMVT_DIR, 'fsaverage', 'labels', 'labels_data'))
    data_var, outliers, region_names, shapiro_p_vals = [], [], [], []
    for region, region_values in all_regions_values.items():
        region_names.append(get_aparc_label_name(region))
        data_var.append(region_values.var())
        outliers.append((len(np.where(region_values > high_threshold)[0]) * 100) / len(region_values))
        shapiro_p_vals.append(shapiro(region_values)[1])
    for data, file_name in zip(
            [data_var, outliers, shapiro_p_vals],
            ['CBF_hist_var', 'CBF_hist_outliers', 'CBF_hist_shapiro']):
        np.savez(op.join(fol, '{}.npz'.format(file_name)), names=region_names, atlas='aparc',
                 data=data, title=file_name.replace('_', ' '), data_min=np.min(data), data_max=np.max(data),
                 cmap='YlOrRd')

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


def run_function_in_parallel(func, all_params, n_jobs, split_jobs=True):
    # Filter our items that should not run (all input files should exist, and all output files shoud not)
    params = [p for p in all_params if all(_run_func_in_parallel((func, [p], True, 0)))]
    print('*** Run {}/{} records, {} jobs ***'.format(len(params), len(all_params), n_jobs))
    ret = input('ok? (y/n)')
    if not au.is_true(ret):
        return
    if split_jobs:
        chunks_indices = np.array_split(np.arange(len(params)), n_jobs)
        chunk_params = [(func, [params[ind] for ind in chunk_indices], False, thread_ind)
                        for thread_ind, chunk_indices in enumerate(chunks_indices)]
    else:
        chunk_params = [(func, [p], False, thread_ind) for thread_ind, p in enumerate(params)]
    utils.run_parallel(_run_func_in_parallel, chunk_params, n_jobs)


def _run_func_in_parallel(parallel_params):
    func, params, check_is_should_run, thread_ind = parallel_params
    flags = []
    now = time.time()
    for run_num, p in enumerate(params):
        if not check_is_should_run:
            utils.time_to_go(now, run_num, len(params), 1, thread_ind)
        subject, site, subject_dir, scan_rescan = [p[key] for key in ['subject', 'site', 'subject_dir', 'scan_rescan']]
        overwrite, print_only = [p.get(key, False) for key in ['overwrite', 'print_only']]
        flag = False
        if func.__name__ == 'register_cbf_to_t1':
            flag = register_cbf_to_t1(subject, subject_dir, scan_rescan,  overwrite, print_only, check_is_should_run)
        flags.append(flag)
    return flags


if __name__ == '__main__':
    subject = os.environ['SUBJECT'] = '277S0203'
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
    sites = ['131-NeuroBeh_ACH', '277-NDC', '800-Hoglund', '829-EmoryUniversity', '960-VitalImaging']
    months = [''] #, '_6', '_12']
    atlas = 'aparc' # 'aparc.DKTatlas'
    low_threshold, high_threshold = 0, 100
    cortex_frac_threshold = 0.9
    overwrite = True
    print_only = False
    do_plot = False
    n_jobs = max(utils.get_n_jobs(30), 4)
    print('n_jobs: {}'.format(n_jobs))

    # read_hippocampus_volumes()
    # calc_volume_fractions_all_subjects(subjects, site, overwrite, print_only)

    params = []
    for site in sites:
        subjects = get_subjects(site)
        good_subjects = preproc_anat(subjects, overwrite_files=False)
        for sub_ind, subject in enumerate(good_subjects):
            # utils.time_to_go(now, sub_ind, len(subjects), 1)
            # get_labels_data(subject, atlas)
            subjects_0_6_12_dirs = [
                op.join(HOME_FOL, site, '{}{}'.format(subject, month)) for month in months
                if op.isdir(op.join(HOME_FOL, site, '{}{}'.format(subject, month)))]
            for subject_dir in subjects_0_6_12_dirs:
                for scan_rescan in [RESCAN]:# SCAN_RESCAN:
                    params.append(vars_names_to_dict((
                        'subject', 'site', 'subject_dir', 'scan_rescan', 'overwrite', 'print_only'), locals()))
                    # calc_volume_fractions(subject, site, subject_dir, scan_rescan, overwrite=False)
                    # register_cbf_to_t1(subject, site, scan_rescan, overwrite=overwrite, print_only=print_only)
                    # project_cbf_on_cortex(subject, site, scan_rescan, overwrite=overwrite, print_only=print_only)
                    # register_aseg_to_cbf(subject, site, scan_rescan, overwrite=overwrite, print_only=print_only)
                    # calc_cbf_histograms(subject, scan_rescan, low_threshold, high_threshold, cortex_frac_threshold,
                    #                     overwrite=True, do_plot=False)
                    # calc_cortical_histograms(subject, scan_rescan, atlas, low_threshold, high_threshold, overwrite, do_plot)
                    # morph_to_fsaverage(subject, scan_rescan, overwrite=overwrite, print_only=print_only)
                    pass
                # calc_scan_rescan_diff(subject, overwrite=True)
                # find_diff_clusters(subject, atlas='laus125', overwrite=True)
    run_function_in_parallel(register_cbf_to_t1, params, n_jobs, split_jobs=True)


    # calc_scan_rescan_mean_diffs(subjects, do_plot_hist=True, overwrite=True)
    # plot_subjects_cbf_histograms(subjects, site, overwrite=True)
    # plot_registration_cost_hist(subjects, site)
    # check_cbf_to_t1_registeration('277S0229', site, 'scan', print_only)

