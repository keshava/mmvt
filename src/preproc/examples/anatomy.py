import os.path as op
import argparse
import shutil
import os
import glob

from src.preproc import anatomy as anat
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu
# from gooey import Gooey

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def get_subject_files_using_sftp(args):
    for subject in args.subject:
        args = anat.read_cmd_args(dict(
            subject=subject,
            atlas=args.atlas,
            sftp_username=args.sftp_username,
            sftp_domain=args.sftp_domain,
            sftp=True,
            remote_subject_dir=args.remote_subject_dir,
            function='prepare_subject_folder'
        ))
        pu.run_on_subjects(args, anat.main)


def get_subject_files_from_mad(org_args=None, subjects=None, necessary_files=None):
    subjects = org_args.subject if org_args is not None else subjects
    for subject in subjects:
        root_fol = '/mnt/cashlab/Original Data/{}'.format(subject[:2].upper())
        args = anat.read_cmd_args(dict(
            subject=subject,
            atlas=org_args.atlas,
            remote_subject_dir=op.join(root_fol, '{subject}/{subject}_Notes_and_Images/{subject}_SurferOutput'),
            function='prepare_subject_folder'
        ))
        if necessary_files is not None:
            args.necessary_files = necessary_files
        pu.run_on_subjects(args, anat.main)


@utils.check_for_freesurfer
def create_annot_from_mad(args):
    remote_subject_dir_template = '/mnt/cashlab/Original Data/MG/{subject}/{subject}_Notes_and_Images/{subject}_SurferOutput'
    for subject in args.subject:
        remote_subject_dir = remote_subject_dir_template.format(subject=subject)
        if utils.both_hemi_files_exist(op.join(remote_subject_dir, 'label', '{hemi}.aparc.DKTatlas.annot')):
            print('{} has already both annot files!'.format(subject))
            continue
        args = anat.read_cmd_args(dict(
            subject=subject.lower(),
            atlas=args.atlas,
            remote_subject_dir=remote_subject_dir_template,
            function='create_annotation',
            ignore_missing=True,
        ))
        pu.run_on_subjects(args, anat.main)
        if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject.lower(), 'label', '{hemi}.aparc.DKTatlas.annot')):
            print('Couldn\'t create annot files for {}!'.format(subject))
            continue
        local_annot_fol = utils.make_dir(op.join(SUBJECTS_DIR, 'annot_files', subject.lower()))
        for hemi in utils.HEMIS:
            local_annot_fname = op.join(SUBJECTS_DIR, subject.lower(), 'label', '{}.aparc.DKTatlas.annot'.format(hemi))
            remote_annot_fname = op.join(remote_subject_dir, 'label', '{}.aparc.DKTatlas.annot'.format(hemi))
            local_temp_annot_fname = op.join(local_annot_fol, '{}.aparc.DKTatlas.annot'.format(hemi))
            if not op.isfile(remote_annot_fname):
                if op.isfile(local_annot_fname):
                    utils.copy_file(local_annot_fname, local_temp_annot_fname)
                else:
                    print('Can\'t copy {} for {}, it doesn\'t exist!'.format(local_annot_fname, subject))


# def get_subject_files_from_server(subject, args):
#     args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
#     args.remote_subject_dir = op.join('/autofs/cluster/neuromind/npeled/subjects', subject)
#     pu.run_on_subjects(args, anat.main)


def prepare_subject_folder_from_franklin(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    args.remote_subject_dir = op.join('/autofs/space/franklin_003/users/npeled/subjects_old/{}'.format(subject))
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def darpa(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            remote_subject_dir=op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(darpa_subject))
        ))
        pu.run_on_subjects(args, anat.main)


def darpa_prep(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            function='prepare_subject_folder',
            remote_subject_dir=op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(darpa_subject))
        ))
        pu.run_on_subjects(args, anat.main)


def darpa_prep_angelique(args):
    import glob
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        root = op.join('/homes/5/npeled/space1/Angelique/recon-alls', darpa_subject)
        recon_all_dirs = glob.glob(op.join(root, '**', '*SurferOutput*'), recursive=True)
        if len(recon_all_dirs) == 0:
            print("Can't find the recon-all folder for {}!".format(subject))
            continue
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            function='prepare_subject_folder',
            remote_subject_dir=recon_all_dirs[0]
        ))
        pu.run_on_subjects(args, anat.main)



def darpa_sftp(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            remote_subject_dir=op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(darpa_subject)),
            sftp=True,
            sftp_username='npeled',
            sftp_domain='door.nmr.mgh.harvard.edu',
        ))
        pu.run_on_subjects(args, anat.main)


def darpa_prep_huygens(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    subject = subject[:2].upper() + subject[2:]
    args.remote_subject_dir = op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(subject))
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def darpa_prep_lili(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas, '--sftp_username', args.sftp_username, '--sftp_domain', args.sftp_domain])
    args.remote_subject_dir = op.join('/autofs/space/lilli_001/users/DARPA-Recons', subject)
    args.sftp = True
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def add_parcellation(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    args.function = 'create_annotation_from_template,parcelate_cortex,calc_faces_verts_dic,' + \
        'save_labels_vertices,save_hemis_curv,calc_labels_center_of_mass,save_labels_coloring'
    pu.run_on_subjects(args, anat.main)


def get_subject_files_using_sftp_from_ohad(subject, args):
    args = anat.read_cmd_args(['-s', subject,'-a', args.atlas])
    args.sftp = True
    args.sftp_username = 'ohadfel'
    args.sftp_domain = '127.0.0.1'
    args.sftp_port = 4444
    args.sftp_subject_dir = '/media/ohadfel/New_Volume/subs/{}'.format(subject)
    args.remote_subject_dir = '/media/ohadfel/New_Volume/subs/{}'.format(subject)
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def get_subject_files_from_server(args):
    args = anat.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='prepare_subject_folder',
        sftp=True,
        sftp_username='npeled',
        sftp_domain='door.nmr.mgh.harvard.edu',
        remote_subject_dir='/space/thibault/1/users/npeled/subjects/{subject}'))
    pu.run_on_subjects(args, anat.main)


def recon_all(args):
    # python -m src.preoroc.anatomy -f recon-all --ignore_missing 1 --n_jobs 1
    # --nifti_fname "/autofs/space/thibault_001/users/npeled/T1/{subject}/mprage.nii" -s "wake5,wake6,wake7,wake8"
    args = anat.read_cmd_args(dict(
        subject=args.subject,
        function='recon-all',
        nifti_fname='/autofs/space/thibault_001/users/npeled/T1/{subject}/mprage.nii',
        ignore_missing=True,
        n_jobs=1,
    ))
    pu.run_on_subjects(args, anat.main)



def pre_meg_coregistration(subject):
    # python -m src.preproc.anatomy -s nmr01391 -f create_outer_skin_surface,check_bem
    # setenv SUBJECT subject
    # mne_setup_mri
    # cd raw_folder
    # mne_analyze: 1) load pial 2) load dig points 3) adjust -> coordinate alignment, 4) viewer window
    # 5) set fiducials, 6) Options -> Show digitizer data 7) Align using fiducials 8) ICP 10 steps
    # 9) If satisfied, press Save mri set in the Adjust coordinate alignment window.
    pass


@utils.check_for_freesurfer
@utils.check_for_mne
def recon_all_clin(args):
    # python -m src.preproc.examples.anatomy -s nmr01426 -f recon_all_clin --clin_fol clin_6966926 --dicoms_fol Prisma_fit-67026-20200618-141203-000586
    import os
    for subject, clin_fol, dicoms_fol in zip(args.subject, args.clin_fol, args.dicoms_fol):
        clin_full_fol = utils.make_dir(op.join(args.clin_root, clin_fol, 'mne_dicom'))
        memprage_fols = glob.glob(op.join(clin_full_fol, '*MEMPRAGE*'))
        print('mne_organize_dicom output fol: {}'.format(clin_full_fol))
        if len(memprage_fols) > 0 :
            ret = au.is_true(input('It seems like you already have memprage folders, are you sure you want to rerun?'))
            if not ret:
                continue
            utils.delete_folder_files(clin_full_fol)
        fs_dir = utils.make_dir(op.join(args.fs_root, subject))
        print('FreeSurfer output fol: {}'.format(fs_dir))
        dicoms_full_path = op.join(args.dicoms_root, dicoms_fol)
        if not op.isdir(dicoms_full_path):
            print('{} does not exist!'.format(dicoms_full_path))
            continue
        rs = utils.partial_run_script(locals(), print_only=args.print_only)
        os.chdir(clin_full_fol)
        rs('mne_organize_dicom {dicoms_full_path}')
        anat.recon_all(subject, clin_full_fol, overwrite=True, subjects_dir=args.fs_root,
              print_only=False, n_jobs=args.n_jobs)
        args = anat.read_cmd_args(dict(
            subject=subject,
            function='all,create_skull_surfaces',
            remote_subject_dir = op.join(args.fs_root, subject),
            n_jobs=args.n_jobs,
        ))
        pu.run_on_subjects(args, anat.main)


@utils.check_for_freesurfer
@utils.check_for_mne
def anat_preproc_clin(args):
    # python -m src.preproc.examples.anatomy -s nmr01426 -f anat_preproc_clin
    for subject in args.subject:
        args = anat.read_cmd_args(dict(
            subject=subject,
            function='all,create_outer_skin_surface,check_bem',
            remote_subject_dir = op.join(args.fs_root, subject),
            n_jobs=args.n_jobs,
        ))
        pu.run_on_subjects(args, anat.main)


# https://github.com/chriskiehl/Gooey
# @Gooey
def main():
    import collections
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-u', '--sftp_username', help='sftp username', required=False, default='npeled')
    parser.add_argument('-d', '--sftp_domain', help='sftp domain', required=False, default='door.nmr.mgh.harvard.edu')
    parser.add_argument('--remote_subject_dir', help='remote_subjects_dir', required=False,
                        default='/space/thibault/1/users/npeled/subjects/{subject}')
    parser.add_argument('--fs_root', help='freesurfer_root', required=False,
                        default='/space/megraid/clinical/MEG-MRI/seder/freesurfer')
    parser.add_argument('--dicoms_root', help='dicoms_root', required=False,
                        default='/cluster/archive/331/siemens')
    parser.add_argument('--dicoms_fol', help='dicoms_fol', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--clin_fol', help='clin_fol', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--clin_root', help='clin_fol', required=False, default='/space/megraid/clinical/MEG-MRI/')
    parser.add_argument('--print_only', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--n_jobs', help='jobs num', required=False, default=4, type=int)
    # choices=[f_name for f_name, f in globals().items() if isinstance(f, collections.Callable)
    #                                  if f_name not in ['Gooey', 'main']]
    args = utils.Bag(au.parse_parser(parser))
    # for subject in args.subject:
    globals()[args.function](args)


if __name__ == '__main__':
    main()