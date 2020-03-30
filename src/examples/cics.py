import os.path as op
from src.utils import utils

FS_ROOT = '/autofs/space/nihilus_001/CICS/Longitudinal_processing/baseline_6_12month_longitudinal_recons'
HOME_FOL = '/autofs/space/nihilus_001/CICS/users/noam/CICS/'

mri_robust_register = 'mri_robust_register --mov "{source_fname}" --dst "{target_fname}" --lta {lta_fname} ' + \
                      '--satit --cost {cost_function}'


def register_cbf_to_t1(subject, site, cost_function='nmi', print_only=False):
    for subfol in ['scan', 'rescan']:
        subject_fol = op.join(HOME_FOL, site, subject, subfol)
        source_fname = op.join(subject_fol, 'Control.nii')
        if not op.isfile(source_fname):
            print('The source ({}) does not exist!'.format(source_fname))
            continue
        target_fname = op.join(FS_ROOT, '{0}_recon.long.{0}-base'.format(subject), 'mri', 'T1.mgz')
        if not op.isfile(target_fname):
            print('The target ({}) does not exist!'.format(target_fname))
            continue
        lta_fname = op.join(subject_fol, 'control_to_T1.lta')
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(mri_robust_register)


if __name__ == '__main__':
    subject = '277S0203'
    site = '277-NDC'
    register_cbf_to_t1(subject, site)