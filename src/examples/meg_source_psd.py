import os.path as op
import os
import glob
import numpy as np
from src.utils import utils
from src.preproc import anatomy as anat
from src.preproc import meg
from src.preproc import fMRI as fmri
from src.preproc import connectivity
from src.utils import freesurfer_utils as fu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def calc_meg_source_psd(args):
    subjects = args.subject
    for subject in subjects:
        args.subject = subject
        local_raw_fname = op.join(MEG_DIR, args.task, subject, args.raw_template.format(
            subject=subject, task=args.task))
        remote_raw_fname = op.join(args.remote_root_dir, 'raw_preprocessed', subject, args.raw_template.format(
            subject=subject, task=args.task))
        if not op.isfile(remote_raw_fname):
            print('Can\'t find remote raw file! {}'.format(remote_raw_fname))
            continue
        if op.isfile(local_raw_fname):
            os.remove(local_raw_fname)
        utils.make_link(remote_raw_fname, local_raw_fname)
        if not op.islink(local_raw_fname):
            print('Can\'t create a link to the remote raw!')
            continue
        inv_fname = op.join(MEG_DIR, args.task, subject, args.inv_template.format(subject=subject, task=args.task))
        _args = meg.read_cmd_args(dict(
            subject=subject, mri_subject=subject,
            function='make_forward_solution,calc_inverse_operator,calc_vertices_data_power_bands', #calc_labels_power_spectrum',
            task=args.task,
            data_per_task=True,
            fmin=65, fmax=120,
            raw_fname=local_raw_fname,
            inv_fname=inv_fname,
            remote_subject_dir=args.remote_subject_dir,
            overwrite_labels_power_spectrum=True,
            n_jobs=args.n_jobs
        ))
        meg.call_main(_args)



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
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.subject = pu.decode_subjects(args.subject, remote_subject_dir=args.remote_subject_dir)
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
    print('Done!')