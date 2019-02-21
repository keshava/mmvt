import glob
import os.path as op
import importlib
import sys
import nibabel as nib
import numpy as np

from src.utils import utils
from src.utils import preproc_utils as pu

mmvt_code_fol = utils.get_mmvt_code_root()
ela_code_fol = op.join(utils.get_parent_fol(mmvt_code_fol), 'electrodes_rois')
if ela_code_fol not in sys.path:
    sys.path.append(ela_code_fol)

from find_rois import find_rois
from find_rois import freesurfer_utils as fu
importlib.reload(find_rois)

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

mri_robust_register = 'mri_robust_register --mov {subjects_dir}/{subject_from}/mri/T1.mgz --dst {subjects_dir}/{subject_to}/mri/T1.mgz --lta {subjects_dir}/{subject_from}/mri/{lta_name}.lta --satit --mapmov {subjects_dir}/{subject_from}/mri/T1_to_{subject_to}.mgz --cost nmi'


def init(subject, atlas, n_jobs):
    labels = find_rois.read_labels_vertices(SUBJECTS_DIR, subject, atlas, n_jobs)
    aseg_atlas_fname = op.join(SUBJECTS_DIR, subject, 'mri', '{}+aseg.mgz'.format(atlas))
    aseg_data = nib.load(aseg_atlas_fname).get_data()
    lut_atlast_fname = op.join(SUBJECTS_DIR, subject, 'mri', '{}ColorLUT.txt'.format(atlas))
    lut = fu.import_freesurfer_lut(lut_atlast_fname)
    pia_verts = {}
    for hemi in ['rh', 'lh']:
        pia_verts[hemi], _ = nib.freesurfer.read_geometry(
            op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
    return labels, aseg_data, lut, pia_verts


def calc_elas(subject, specific_elecs_names, template, template_header, bipolar=False,  atlas='aparc.DKTatlas', n_jobs=1):
    cmd_args = ['-s', subject, '-a', atlas, '-b', str(bipolar), '--n_jobs', str(n_jobs)]
    args = find_rois.get_args(cmd_args)
    elecs_names, elecs_pos, elecs_dists, elecs_types, _ = find_rois.get_electrodes(subject, bipolar, args)
    elecs_oris = find_rois.get_electrodes_orientation(elecs_names, elecs_pos, bipolar, elecs_types)
    elecs_info = [(elec_name, elec_pos, elec_dist, elec_type, elec_ori) for
                  elec_name, elec_pos, elec_dist, elec_type, elec_ori in \
                  zip(elecs_names, elecs_pos, elecs_dists, elecs_types, elecs_oris)
                  if elec_name in specific_elecs_names]
    labels, aseg_data, lut, pia_verts = init(subject, atlas, n_jobs)
    len_lh_pia = len(pia_verts['lh'])
    subject_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).header

    for elec_name, elec_pos, elec_dist, elec_type, elec_ori in elecs_info:
        elec_labeling = calc_ela(
            subject, bipolar, elec_name, elec_pos, elec_type, elec_ori, elec_dist, labels, aseg_data, lut,
            pia_verts, len_lh_pia, error_radius=3, elc_length=4, overwrite=False, n_jobs=n_jobs)
        template_pos = calc_template_elec_pos(subject, template, elec_pos, subject_header, template_header)
        elec_labeling_template = calc_ela(
            template, bipolar, elec_name, template_pos, elec_type, elec_ori, elec_dist, labels, aseg_data, lut,
            pia_verts, len_lh_pia, error_radius=3, elc_length=4, overwrite=False, n_jobs=n_jobs)
        comp_elecs_labeling(elec_labeling, elec_labeling_template, elec_name)


def comp_elecs_labeling(elc_labeling, elc_labeling_template, elec_name):
    for roi, prob in zip(elc_labeling['cortical_rois'], elc_labeling['cortical_probs']):
        no_err, err = compare_rois_and_probs(
            subject, template, elec_name, roi, prob, elc_labeling['cortical_rois'],
            elc_labeling_template['cortical_rois'], elc_labeling_template['cortical_probs'])
        if not no_err:
            print(err)
    for roi, prob in zip(elc_labeling['subcortical_rois'], elc_labeling['subcortical_probs']):
        no_err, err = compare_rois_and_probs(
            subject, template, elec_name, roi, prob, elc_labeling['subcortical_rois'],
            elc_labeling_template['subcortical_rois'], elc_labeling_template['subcortical_probs'])
        if not no_err:
            print(err)


def calc_ela(subject, bipolar, elec_name, elec_pos, elec_type, elec_ori, elec_dist, labels, aseg_data, lut, pia_verts,
             len_lh_pia, error_radius=3, elc_length=4, overwrite=False, n_jobs=1):
    strech_to_dist = False
    enlarge_if_no_hit = True
    nei_dimensions = None
    ret_val = find_rois.identify_roi_from_atlas_per_electrode(
        labels, elec_pos, pia_verts, len_lh_pia, lut,
        aseg_data, elec_name, error_radius, elc_length, nei_dimensions, elec_ori, elec_dist, elec_type, strech_to_dist,
        enlarge_if_no_hit, bipolar, SUBJECTS_DIR, subject, excludes=None, n_jobs=1)

    (regions, regions_hits, subcortical_regions, subcortical_hits, approx_after_strech, elc_length, elec_hemi_vertices,
     elec_hemi_vertices_dists, hemi) = ret_val
    regions_probs = np.hstack((regions_hits, subcortical_hits)) / float(
        np.sum(regions_hits) + np.sum(subcortical_hits))
    ret = {'name': elec_name, 'cortical_rois': regions, 'subcortical_rois': subcortical_regions,
           'cortical_probs': regions_probs[:len(regions)],
           'subcortical_probs': regions_probs[len(regions):],
           # 'cortical_indices': elec_hemi_vertices, 'cortical_indices_dists':elec_hemi_vertices_dists,
            'hemi': hemi}
    return ret


def calc_template_elec_pos(subject, template, elec_pos, subject_header=None, template_header=None):
    if subject_header is None:
        subject_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).get_header()
    if template_header is None:
        template_header = nib.load(op.join(SUBJECTS_DIR, template, 'mri', 'T1.mgz')).get_header()
    vox = utils.apply_trans(np.linalg.inv(subject_header.get_vox2ras_tkr()), elec_pos)
    ras = utils.apply_trans(subject_header.get_vox2ras(), vox)
    template_vox = utils.apply_trans(template_header.get_ras2vox(), ras)
    template_elec_pos = utils.apply_trans(template_header.get_vox2ras_tkr(), template_vox)
    return template_elec_pos


def compare_electrodes_labeling(subject, electrodes_names, template, atlas='aparc.DKTatlas'):
    # template_elab_files = glob.glob(op.join(
    #     MMVT_DIR, template, 'electrodes', '{}_{}_electrodes_cigar_r_3_l_4.pkl'.format(template, atlas)))
    # if len(template_elab_files) == 0:
    #     print('No electrodes labeling file for {}!'.format(template))
    #     return
    # elab_template = utils.load(template_elab_files[0])
    errors = ''
    elab_files = glob.glob(op.join(
        MMVT_DIR, subject, 'electrodes', '{}_{}_electrodes_cigar_r_3_l_4.pkl'.format(subject, atlas)))
    if len(elab_files) == 0:
        print('No electrodes labeling file for {}!'.format(subject))
        return
    elab = utils.load(elab_files[0])
    # elab = [e for e in elab if e['name'] in electrodes_names]
    for elc in electrodes_names:
        no_errors = True
        elc_labeling = [e for e in elab if e['name'] == elc][0]
        elc_labeling_template = [e for e in elab_template if e['name'] == '{}_{}'.format(subject, elc)][0]
        for roi, prob in zip(elc_labeling['cortical_rois'], elc_labeling['cortical_probs']):
            no_err, err = compare_rois_and_probs(
                subject, template, elc, roi, prob, elc_labeling['cortical_rois'],
                elc_labeling_template['cortical_rois'], elc_labeling_template['cortical_probs'])
            no_errors = no_errors and no_err
            if err != '':
                errors += err + '\n'
        for roi, prob in zip(elc_labeling['subcortical_rois'], elc_labeling['subcortical_probs']):
            no_err, err = compare_rois_and_probs(
                subject, template, elc, roi, prob, elc_labeling['subcortical_rois'],
                elc_labeling_template['subcortical_rois'], elc_labeling_template['subcortical_probs'])
            no_errors = no_errors and no_err
            if err != '':
                errors += err + '\n'
        if no_errors:
            print('{},{},Good!'.format(subject, elc))
            errors += '{},{},Good!\n'.format(subject, elc)


def compare_rois_and_probs(subject, template, elc, roi, prob, elc_labeling_rois, elc_labeling_template_rois,
                           elc_labeling_template_rois_probs):
    no_errors = True
    err = ''
    if roi not in elc_labeling_template_rois:
        if prob > 0.05:
            err = f'{subject},{elc},{roi} ({prob}) not in {template}'
            print(err)
            no_errors = False
    else:
        roi_ind = elc_labeling_template_rois.index(roi)
        template_roi_prob = elc_labeling_template_rois_probs[roi_ind]
        if abs(prob - template_roi_prob) > 0.05:
            err = f'{subject},{elc},{roi} prob ({prob} != {template} prob ({template_roi_prob})'
            print(err)
            no_errors = False
    for roi, prob in zip(elc_labeling_template_rois, elc_labeling_template_rois_probs):
        if roi not in elc_labeling_rois and prob > 0.05:
            err = f'{subject},{elc},{roi} ({prob}) only in {template}'
            print(err)
            no_errors = False
    return no_errors, err


def robust_register_to_template(subject, subject_to, subjects_dir, vox2vox=False, print_only=False):
    cmd = mri_robust_register
    lta_name = 't1_to_{}'.format(subject_to)
    if vox2vox:
        cmd += ' --vox2vox'
        lta_name += '_vox2vox'
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(cmd)


if __name__ == '__main__':

    subject = 'mg112'
    elec_name = ['RPT1']
    template = 'colin27'
    atlas = 'aparc.DKTatlas40'
    template_header = nib.load(op.join(SUBJECTS_DIR, template, 'mri', 'T1.mgz')).header

    calc_elas(subject, elec_name, template, template_header, bipolar=False, atlas=atlas, n_jobs=1)
    # compare_electrodes_labeling(subject, electrodes_names, template, atlas)