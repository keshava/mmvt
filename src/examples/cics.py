import os.path as op
import os
from src.utils import utils

FS_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_6_12month_longitudinal_recons'
HOME_FOL = '/autofs/space/nihilus_001/CICS/users/noam/CICS/'

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
print('Setting SUBJECTS_DIR to {}'.format(SUBJECTS_DIR))
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR

mri_robust_register = 'mri_robust_register --mov "{source_fname}" --dst "{target_fname}" --lta "{lta_fname}" ' + \
                      '--satit --mapmov "{output_fname}" --cost {cost_function}'
register_using_lta = 'mri_convert -at "{lta_fname}" "{source_fname}" "{output_fname}"'


def register_cbf_to_t1(subject, site, scan_rescan, cost_function='nmi', overwrite=False, print_only=False):
    subject_fol = op.join(HOME_FOL, site, subject, scan_rescan)
    output_fname = op.join(subject_fol, 'Control_to_T1.nii')
    source_fname = op.join(subject_fol, 'Control.nii')
    target_fname = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    if not op.isfile(source_fname):
        print('The source ({}) does not exist!'.format(source_fname))
        return False
    if not op.isfile(target_fname):
        print('The target ({}) does not exist!'.format(target_fname))
        return False
    if not op.isfile(lta_fname) or not op.isfile(output_fname) or overwrite:
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(mri_robust_register)
    if op.isfile(lta_fname):
        register_cbf_using_lta(subject_fol, print_only)
    else:
        print('No registration file found! ({})'.format(lta_fname))
    print_freeview_cmd(subject, subject_fol)


def register_cbf_using_lta(subject_fol, overwrite=False, print_only=False):
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    source_fname = op.join(subject_fol, 'CBF.nii')
    output_fname = op.join(subject_fol, 'CBF_to_T1.nii')
    if not op.isfile(output_fname) or overwrite:
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(register_using_lta)
    else:
        print('{} already exist'.format(output_fname))


def print_freeview_cmd(subject, subject_fol):
    cbf = op.join(subject_fol, 'CBF_to_T1.nii')
    control = op.join(subject_fol, 'Control_to_T1.nii')
    t1 = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
    print('freeview {} {} {}'.format(cbf, control, t1))


def preproc_anat(subject):
    from src.preproc import anatomy as anat
    args = anat.read_cmd_args(dict(
        subject=subject,
        remote_subject_dir=op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject)),
        # exclude='create_new_subject_blend_file',
        # ignore_missing=True
    ))
    anat.call_main(args)


def project_cbf_on_cortex(subject, site, scan_rescan, overwrite=False):
    from src.preproc import fMRI
    fmri_fol = utils.make_dir(op.join(FMRI_DIR, subject))
    mmvt_cbf_fname = op.join(fmri_fol, 'CBF_{}.nii'.format(scan_rescan))
    cics_cbf_fname = op.join(HOME_FOL, site, subject, scan_rescan, 'CBF_to_T1.nii')
    if not op.islink(mmvt_cbf_fname) and op.isfile(cics_cbf_fname):
        utils.make_link(cics_cbf_fname, mmvt_cbf_fname)
    if not op.islink(mmvt_cbf_fname):
        print('Cannot file the link to CBF! ({}-{})'.format(cics_cbf_fname, mmvt_cbf_fname))
        return False
    args = fMRI.read_cmd_args(dict(
        subject=subject,
        function='project_volume_to_surface',
        fmri_file_template=utils.namebase_with_ext(mmvt_cbf_fname),
        overwrite_surf_data=overwrite))
    fMRI.call_main(args)


def calc_scan_rescan_diff(subject, overwrite=False):
    from src.preproc import fMRI
    args = fMRI.read_cmd_args(dict(
        subject=subject, function='calc_files_diff', fmri_file_template= 'CBF_scan,CBF_rescan',
        overwrite_surf_data=overwrite))
    fMRI.call_main(args)


if __name__ == '__main__':
    subject = '277S0203'
    site = '277-NDC'
    overwrite = False
    # preproc_anat(subject)
    for scan_rescan in ['scan', 'rescan']:
        # register_cbf_to_t1(subject, site, scan_rescan)
        # project_cbf_on_cortex(subject, site, scan_rescan, overwrite)
        calc_scan_rescan_diff(subject, scan_rescan, overwrite)