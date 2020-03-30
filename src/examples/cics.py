import os.path as op
from src.utils import utils

FS_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_6_12month_longitudinal_recons'
HOME_FOL = '/autofs/space/nihilus_001/CICS/users/noam/CICS/'

mri_robust_register = 'mri_robust_register --mov "{source_fname}" --dst "{target_fname}" --lta "{lta_fname}" ' + \
                      '--satit --mapmov "{output_fname}" --cost {cost_function}'
register_using_lta = 'mri_convert -at "{lta_fname}" "{source_fname}" "{output_fname}"'


def register_cbf_to_t1(subject, site, cost_function='nmi', overwrite=False, print_only=False):
    for subfol in ['scan', 'rescan']:
        subject_fol = op.join(HOME_FOL, site, subject, subfol)
        output_fname = op.join(subject_fol, 'Control_to_T1.nii')
        source_fname = op.join(subject_fol, 'Control.nii')
        target_fname = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
        lta_fname = op.join(subject_fol, 'control_to_T1.lta')
        if not op.isfile(source_fname):
            print('The source ({}) does not exist!'.format(source_fname))
            continue
        if not op.isfile(target_fname):
            print('The target ({}) does not exist!'.format(target_fname))
            continue
        if not op.isfile(lta_fname) or not op.isfile(output_fname) or overwrite:
            rs = utils.partial_run_script(locals(), print_only=print_only)
            rs(mri_robust_register)
        if op.isfile(lta_fname):
            register_cbf_using_lta(subject_fol, print_only)
        else:
            print('No registration file found! ({})'.format(lta_fname))


def register_cbf_using_lta(subject_fol, overwrite=False, print_only=False):
    lta_fname = op.join(subject_fol, 'control_to_T1.lta')
    source_fname = op.join(subject_fol, 'CBF.nii')
    output_fname = op.join(subject_fol, 'CBF_to_T1.nii')
    if not op.isfile(output_fname) or overwrite:
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(register_using_lta)
    else:
        print('{} already exist'.format(output_fname))


if __name__ == '__main__':
    subject = '277S0203'
    site = '277-NDC'
    register_cbf_to_t1(subject, site)