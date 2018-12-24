import os.path as op
import os
import glob
import shutil
from src.utils import utils
from src.preproc import meg
from src.preproc import eeg


LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def get_meg_empty_fnames(subject, remote_fol, args, ask_for_different_day_empty=False):
    remote_fol = op.join(remote_fol, subject)
    csv_fname = op.join(remote_fol, 'cfg.txt')
    if not op.isfile(csv_fname):
        print('No cfg file ({})!'.format(csv_fname))
        return '', '', ''
    day, empty_fname, cor_fname, local_rest_raw_fname = '', '', '', ''
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4].lower() == 'resting':
            day = line[2]
            remote_rest_raw_fname = op.join(remote_fol, line[0].zfill(3), line[-1])
            if not op.isfile(remote_rest_raw_fname):
                raise Exception('rest file does not exist! {}'.format(remote_rest_raw_fname))
            break
    if day == '':
        print('Couldn\'t find the resting day in the cfg!')
        return '', '', ''
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4] == 'empty':
            empty_fname = op.join(MEG_DIR, subject, '{}_empty_raw.fif'.format(subject))
            if op.isfile(empty_fname):
                continue
            if line[2] == day:
                remote_empty_fname = op.join(remote_fol, line[0].zfill(3), line[-1])
    if not op.isfile(empty_fname):
        for line in utils.csv_file_reader(csv_fname, ' '):
            if line[4] == 'empty':
                ret = input('empty recordings from a different day were found, continue (y/n)? ') \
                        if ask_for_different_day_empty else True
                if au.is_true(ret):
                    remote_empty_fname = op.join(remote_fol, line[0].zfill(3), line[-1])
    cor_dir = op.join(args.remote_subject_dir.format(subject=subject), 'mri', 'T1-neuromag', 'sets')
    if op.isfile(op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))):
        cor_fname = op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))
    elif op.isfile(op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))):
        cor_fname = op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))
    return local_rest_raw_fname, empty_fname, cor_fname


def read_clin_meg_layouts(args):
    subjects = args.subject
    for subject in subjects:
        remote_subject_dir = find_seder_remote_subject_dir(subject)
        if not op.isdir(remote_subject_dir):
            print('{}: Can\'t find remote_subject_dir!'.format(subject))
        args.remote_subject_dir = remote_subject_dir
        remote_raw_template = op.join(args.raw_clin_rest_remote_fol, '{}*'.format(subject))
        remote_raw_fols = glob.glob(remote_raw_template)
        if len(remote_raw_fols) == 0:
            print('Can\'t find raw fol! {}'.format(remote_raw_template))
            continue
        for remote_raw_fol in remote_raw_fols:
            remote_raw_fnames_template = op.join(remote_raw_fol, '{}*_Resting_eeg_meg_ica-raw.fif'.format(subject))
            remote_raw_fnames = glob.glob(remote_raw_fnames_template)
            if len(remote_raw_fnames) > 0:
                break
        if len(remote_raw_fnames) == 0:
            print('Can\'t find remote_raw_fname! {}'.format(remote_raw_fnames_template))
            continue
        read_meg_layouts(args, remote_raw_fnames[0])
        read_eeg_layouts(args, remote_raw_fnames[0])


def find_seder_remote_subject_dir(subject):
    seder_root = os.environ.get('SEDER_SUBJECT_META', '')
    if seder_root != '':
        remote_subject_dir = glob.glob(op.join(seder_root, subject, 'freesurfer', '*'))[0]
    else:
        remote_subject_dir = [d for d in [
            '/autofs/space/megraid_clinical/MEG-MRI/seder/freesurfer/{}'.format(subject),
            '/home/npeled/subjects/{}'.format(subject),
            op.join(SUBJECTS_DIR, subject)] if op.isdir(d)][0]
    return remote_subject_dir


def read_meg_layouts(args, remote_raw_fname=''):
    bad_subjects = []
    output_fol = utils.make_dir(op.join(MMVT_DIR, 'sensors'))
    subjects = args.subject
    for subject in subjects:
        args.subject = subject
        _, _, trans_fname = get_meg_empty_fnames(subject, args.remote_meg_dir, args)
        if remote_raw_fname == '':
            remote_raw_fname = op.join(args.raw_rest_remote_fol, subject, '{}_Resting_meg_ica-raw.fif'.format(subject))
        if not op.isfile(remote_raw_fname):
            print('No Cor fname: {}!!!'.format(remote_raw_fname))
            continue
        meg_args = meg.read_cmd_args(utils.Bag(
            subject=subject,
            data_per_task=False,
            remote_subject_dir=args.remote_subject_dir,
            raw_fname=remote_raw_fname,
            function='read_sensors_layout',
            trans_fname = trans_fname,
            overwrite_sensors=True,
            read_info_file=False,
            n_jobs=args.n_jobs
        ))
        meg.call_main(meg_args)
        for sensor_type in ['mag', 'planar1', 'planar2']:
            sensors_fname = op.join(MMVT_DIR, subject, 'meg', 'meg_{}_sensors_positions.npz'.format(sensor_type))
            if not op.isfile(sensors_fname):
                bad_subjects.append(subject)
                break
            utils.make_dir(op.join(output_fol, subject))
            output_fname = op.join(output_fol, subject, utils.namebase_with_ext(sensors_fname))
            if not op.isfile(output_fname):
                shutil.copyfile(sensors_fname, output_fname)

    print('{}/{} subjects with no sensors:'.format(len(bad_subjects), len(args.subject)))
    print(bad_subjects)


def read_eeg_layouts(args, remote_raw_fname=''):
    bad_subjects = []
    output_fol = utils.make_dir(op.join(MMVT_DIR, 'sensors'))
    for subject in args.subject:
        _, _, trans_fname = get_meg_empty_fnames(subject, args.remote_meg_dir, args)
        if remote_raw_fname == '':
            remote_raw_fname = op.join(args.raw_rest_remote_fol, subject, '{}_Resting_eeg_ica-raw.fif'.format(subject))
        if not op.isfile(remote_raw_fname):
            print('No Cor fname: {}!!!'.format(remote_raw_fname))
            continue
        eeg_args = eeg.read_cmd_args(utils.Bag(
            subject=subject,
            data_per_task=False,
            remote_subject_dir=args.remote_subject_dir,
            raw_template=remote_raw_fname,
            function='read_sensors_layout',
            trans_fname = trans_fname,
            overwrite_sensors=True,
            read_info_file=False,
            n_jobs=args.n_jobs
        ))
        eeg.call_main(eeg_args)
        sensors_fname = op.join(MMVT_DIR, subject, 'eeg', 'eeg_sensors_positions.npz')
        if not op.isfile(sensors_fname):
            bad_subjects.append(subject)
        else:
            utils.make_dir(op.join(output_fol, subject))
            output_fname = op.join(output_fol, subject, utils.namebase_with_ext(sensors_fname))
            if not op.isfile(output_fname):
                shutil.copyfile(sensors_fname, output_fname)

    print('{}/{} subjects with no sensors:'.format(len(bad_subjects), len(args.subject)))
    print(bad_subjects)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    from src.utils import preproc_utils as pu

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('--raw_rest_remote_fol', required=False,
        default='/autofs/space/karima_002/users/Resting/raw_preprocessed')
    parser.add_argument('--remote_subject_dir', required=False,
        default='/autofs/space/lilli_001/users/DARPA-Recons/{subject}')
    parser.add_argument('--remote_meg_dir', required=False,
        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg')
    parser.add_argument('--raw_clin_rest_remote_fol', required=False,
        default='/autofs/space/karima_002/users/Machine_Learning_Clinical_MEG_EEG_Resting/raw_preprocessed')
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.subject = pu.decode_subjects(args.subject, remote_subject_dir=args.remote_subject_dir)
    locals()[args.function](args)
