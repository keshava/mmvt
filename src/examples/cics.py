import os.path as op
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib

from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.utils import labels_utils as lu
from src.preproc import fMRI
from src.preproc import anatomy as anat

FS_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_recons'
FS_BASE_6_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_6month_recons'
HOME_FOL = '/autofs/space/nihilus_001/CICS/users/noam/CICS/'
RESULTS_FOL = '/autofs/space/nihilus_001/CICS/users/noam/figures/'
SCAN, RESCAN = 'scan', 'rescan'

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


def register_cbf_to_t1(subject, site, scan_rescan, overwrite=False, print_only=False): # cost_function='nmi',
    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    output_fname = op.join(subject_fol, 'Control_to_T1.nii')
    source_fname = op.join(subject_fol, 'Control.nii')
    target_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.mgz')
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    tmp_fol = utils.make_dir(op.join(subject_fol, 'bbregister'))
    if not op.isfile(source_fname):
        print('The source ({}) does not exist!'.format(source_fname))
        return False
    if not op.isfile(target_fname):
        print('The target ({}) does not exist!'.format(target_fname))
        return False
    if not op.isfile(lta_fname) or not op.isfile(output_fname) or overwrite:
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(bbregister)
    else:
        print('Nothing to do, overwrite=False')
    if not op.isfile(lta_fname):
        print('No registration file found! ({})'.format(lta_fname))
    '''
    tkregisterfv --mov /autofs/space/nihilus_001/CICS/users/noam/CICS/277-NDC/277S0203/rescan/Control.nii --reg /autofs/space/nihilus_001/CICS/users/noam/CICS/277-NDC/277S0203/rescan/control_to_T1.lta --surfs
    cd /autofs/space/nihilus_001/CICS/users/noam/mmvt_root/mmvt_code
    freeview -transform-volume -viewport cor -v /autofs/space/nihilus_001/CICS/users/noam/mmvt_root/subjects/277S0203/mri/orig.mgz:visible=0:name=orig.mgz(targ) /autofs/space/nihilus_001/CICS/users/noam/CICS/277-NDC/277S0203/rescan/Control.nii:name=Control.nii(mov):reg=/autofs/space/nihilus_001/CICS/users/noam/CICS/277-NDC/277S0203/rescan/control_to_T1.lta --surface /autofs/space/nihilus_001/CICS/users/noam/mmvt_root/subjects/277S0203/surf/lh.white:edgecolor=yellow --surface /autofs/space/nihilus_001/CICS/users/noam/mmvt_root/subjects/277S0203/surf/rh.white:edgecolor=yellow
    '''


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


def calc_cbf_histograms(subject, scan_rescan, low_threshold, high_threshold, overwrite=False, do_plot=True):
    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    output_fname = op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'aparc_aseg_hist.pkl')
    if op.isfile(output_fname) and not overwrite and not do_plot:
        return True
    cics_cbf_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF.nii')
    if not op.isfile(cics_cbf_fname):
        print('calc_cbf_histograms: Cannot find {}!'.format(cics_cbf_fname))
        return False
    aparc_aseg_fname = op.join(subject_fol, 'aparc+aseg_cbf.mgz')
    if not op.isfile(aparc_aseg_fname):
        print('calc_cbf_histograms: Cannot find {}!'.format(aparc_aseg_fname))
        return False
    cbf_data = nib.load(cics_cbf_fname).get_data()
    lut = utils.read_freesurfer_lookup_table(return_dict=True)
    aparc_aseg = nib.load(aparc_aseg_fname).get_data()
    unique_codes = list(range(1001, 1036)) + list(range(2001, 2036))
    output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'aparc_aseg'))
    if overwrite and do_plot:
        utils.delete_folder_files(output_fol)
    regions_values = {}
    for code in tqdm(unique_codes):
        region_name = lut.get(code, None)
        if region_name is None:
            # print('{} not in lut!'.format(code))
            continue
        fig_fname = op.join(output_fol, '{}.jpg'.format(region_name))
        if op.isfile(fig_fname) and not overwrite:
            continue
        region_values = cbf_data[np.where(aparc_aseg == code)]
        regions_values[region_name] = region_values.copy()
        outliers = np.where(region_values < low_threshold)[0]
        if len(outliers) > 0:
            print('{} has {} out of {} values < {}'.format(region_name, len(outliers), len(region_values), low_threshold))
        outliers = np.where(region_values > high_threshold)[0]
        if len(outliers) > 0:
            print('{} has {} out of {} values > {}'.format(region_name, len(outliers), len(region_values), high_threshold))
        region_values[region_values < low_threshold] = low_threshold
        region_values[region_values > high_threshold] = high_threshold
        if do_plot:
            plt.hist(region_values, bins=40)
            plt.savefig(fig_fname)
            plt.close()
    utils.save(regions_values, output_fname)


# def print_freeview_cmd(subject, subject_fol):
#     cbf = op.join(subject_fol, 'CBF_to_T1.nii')
#     control = op.join(subject_fol, 'Control_to_T1.nii')
#     t1 = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
#     print('freeview {} {} {}'.format(cbf, control, t1))

def get_labels_data(subject, atlas):
    necessary_files = {'label': ['{}.{}.annot'.format(hemi, atlas) for hemi in utils.HEMIS]}
    remote_subject_dir = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject))
    utils.prepare_subject_folder(necessary_files, subject, remote_subject_dir, SUBJECTS_DIR)


def preproc_anat(subject):
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    utils.make_dir(op.join(MMVT_DIR, '{}_rescan'.format(subject), 'fmri'))

    args = anat.read_cmd_args(dict(
        subject=subject, remote_subject_dir=op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject)),
        function='prepare_subject_folder',
        exclude='create_new_subject_blend_file', ignore_missing=True))
    anat.call_main(args)

    # args = anat.read_cmd_args(dict(
    #     subject=subject, remote_subject_dir=op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject)),
    #     function='labeling', atlas='laus125', ignore_missing=True))
    # anat.call_main(args)

    args = anat.read_cmd_args(dict(
        subject='{}_rescan'.format(subject),
        remote_subject_dir=op.join(FS_ROOT, '{0}_B_recon.long.{0}-base'.format(subject)),
        function='prepare_subject_folder',
        exclude='create_new_subject_blend_file', ignore_missing=True))
    anat.call_main(args)


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


def calc_scan_rescan_diff(subject, overwrite=False):
    fMRI.calc_files_diff(
        subject, 'fmri_CBF_scan_{hemi},fmri_CBF_rescan_{hemi}', 'CBF_scan_rescan', 'zvals', overwrite)


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


def calc_volume_fractions(subject, site, scan_rescan, use_reg=True, overwrite=False):
    # 'mri_compute_volume_fractions --o "{output_fname}" --regheader {subject} "{target_fname}"'
    # mri_compute_volume_fractions_reg = 'mri_compute_volume_fractions --o "{output_fname}" --reg  "{reg_fname}"'
    if use_reg:
        reg_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'control_to_T1.lta')
        output_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF')
        rs = utils.partial_run_script(locals(), print_only=print_only)
        output_full_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF.cortex.mgz')
    else:
        output_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'T1')
        target_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.mgz')
        rs = utils.partial_run_script(locals(), print_only=print_only)
        output_full_fname = op.join(FS_ROOT, get_subject_fs_folder(subject, scan_rescan), 'mri', 'T1.cortex.mgz')
    if not op.isfile(output_full_fname) or overwrite:
        rs(mri_compute_volume_fractions_reg if use_reg else mri_compute_volume_fractions)


def plot_subjects_cbf_histograms(subjects, site, overwrite=False):
    from collections import defaultdict

    output_fol = utils.make_dir(op.join(RESULTS_FOL, site, 'hists', 'cbf_aparc_hists'))
    output_fname = op.join(RESULTS_FOL, site, 'cbf_aparc_hists.pkl')
    x_grid = np.linspace(-50, 150, 200)
    if not op.isfile(output_fname) or overwrite:
        kdes = defaultdict(list)
        for subject in tqdm(subjects):
            for scan_rescan in [SCAN, RESCAN]:
                input_fname = op.join(MMVT_DIR, subject, 'ASL', scan_rescan, 'aparc_aseg_hist.pkl')
                if not op.isfile(input_fname):
                    print('No hist file for {} {}'.format(subject, scan_rescan))
                    continue
                regions_values = utils.load(input_fname)
                for region, values in regions_values.items():
                    kdes[region].append(utils.kde(values, x_grid, bandwidth=0.1))
        utils.save(kdes, output_fname)
    else:
        kdes = utils.load(output_fname)

    for region_name, region_kdes in tqdm(kdes.items()):
        figure_fname = op.join(output_fol, '{}.jpg'.format(region_name))
        if op.isfile(figure_fname) and not overwrite:
            continue
        plt.figure()
        for region_kde in region_kdes:
            plt.plot(x_grid, region_kde)
        plt.savefig(figure_fname)
        plt.close()


def get_subjects(site):
    # subjects = set([fol.split('_')[0].replace('-base', '') for fol in utils.get_subfolders(FS_ROOT, 'name')])
    # return sorted(list(subjects - set(['scripts', 'fsaverage'])))
    subjects = set([fol.split('_')[0] for fol in utils.get_subfolders(op.join(HOME_FOL, site), 'name')])
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


if __name__ == '__main__':
    subject = os.environ['SUBJECT'] = '277S0203'
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
    site = '277-NDC'
    atlas = 'aparc' # 'aparc.DKTatlas'
    low_threshold, high_threshold = 0, 100
    overwrite = False
    print_only = False
    do_plot = False
    subjects = get_subjects(site)
    # subjects = [subject]

    # read_hippocampus_volumes()

    now = time.time()
    for sub_ind, subject in enumerate(subjects):
        utils.time_to_go(now, sub_ind, len(subjects), 1)
        # preproc_anat(subject)
        # get_labels_data(subject, atlas)
        for scan_rescan in [SCAN, RESCAN]:
            calc_volume_fractions(subject, site, scan_rescan)
            # register_cbf_to_t1(subject, site, scan_rescan, overwrite=overwrite, print_only=print_only)
            # project_cbf_on_cortex(subject, site, scan_rescan, overwrite=overwrite, print_only=print_only)
            # register_aseg_to_cbf(subject, site, scan_rescan, overwrite=overwrite, print_only=print_only)
            # calc_cbf_histograms(subject, scan_rescan, low_threshold, high_threshold, overwrite=True, do_plot=False)
            # calc_cortical_histograms(subject, scan_rescan, atlas, low_threshold, high_threshold, overwrite, do_plot)
            # morph_to_fsaverage(subject, scan_rescan, overwrite=overwrite, print_only=print_only)
            pass
        # calc_scan_rescan_diff(subject, overwrite=overwrite)
        # find_diff_clusters(subject, atlas='laus125', overwrite=True)
    # plot_subjects_cbf_histograms(subjects, site, True)
    # plot_registration_cost_hist(subjects, site)