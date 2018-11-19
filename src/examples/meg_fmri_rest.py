import os.path as op
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


def init_anatomy(args):
    args = anat.read_cmd_args(dict(
        subject=args.subject,
        remote_subject_dir=args.remote_subject_dir,
        exclude='create_new_subject_blend_file',
        ignore_missing=True
    ))
    anat.call_main(args)


def init_meg(subject):
    utils.make_dir(op.join(MEG_DIR, subject))
    utils.make_link(op.join(args.remote_subject_dir.format(subject=subject), 'bem'),
                    op.join(MEG_DIR, subject, 'bem'))
    utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'))


def get_meg_empty_fnames(subject, remote_fol, args):
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
            local_rest_raw_fname = op.join(MEG_DIR, subject, '{}_resting_raw.fif'.format(subject))
            if not op.isfile(local_rest_raw_fname):
                utils.make_link(remote_rest_raw_fname, local_rest_raw_fname)
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
                if not op.isfile(remote_empty_fname):
                    raise Exception('empty file does not exist! {}'.format(remote_empty_fname))
                utils.make_link(remote_empty_fname, empty_fname)
    cor_dir = op.join(args.remote_subject_dir.format(subject=subject), 'mri', 'T1-neuromag', 'sets')
    if op.isfile(op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))):
        cor_fname = op.join(cor_dir, 'COR-{}-resting.fif'.format(subject))
    elif op.isfile(op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))):
        cor_fname = op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, day))
    return local_rest_raw_fname, empty_fname, cor_fname


def get_fMRI_rest_fol(subject, remote_root):
    remote_fol = op.join(remote_root, '{}_01'.format(subject.upper()))
    csv_fname = op.join(remote_fol, 'cfg.txt')
    if not op.isfile(csv_fname):
        print('No cfg file!')
        return '', '', ''
    num = None
    for line in utils.csv_file_reader(csv_fname, '\t'):
        if line[1].lower() == 'resting':
            num = line[0]
            break
    if num is None:
        print ('Can\'t find rest in the cfg file for {}!'.format(subject))
        return ''
    subject_folders = glob.glob(op.join(remote_root, '{}_*'.format(subject.upper())))
    rest_fols = []
    for subject_fol in subject_folders:
        rest_fols = glob.glob(op.join(subject_fol, '**', num.zfill(3)), recursive=True)
        if len(rest_fols) == 1:
            break
    if len(rest_fols) == 0:
        print('Can\'t find rest in the cfg file for {}!'.format(subject))
        return ''
    return rest_fols[0]


def convert_rest_dicoms_to_mgz(subject, rest_fol):
    try:
        root = utils.make_dir(op.join(FMRI_DIR, subject))
        output_fname = op.join(root, '{}_rest.mgz'.format(subject))
        if op.isfile(output_fname):
            return output_fname
        dicom_files = glob.glob(op.join(rest_fol, 'MR*'))
        dicom_files.sort(key=op.getmtime)
        fu.mri_convert(dicom_files[0], output_fname)
        if op.isfile(output_fname):
            return output_fname
        else:
            print('Can\'t find {}!'.format(output_fname))
            return ''
    except:
        utils.print_last_error_line()
        return ''


def analyze_meg(args):
    subjects = args.subject
    for subject, mri_subject in zip(subjects, args.mri_subject):
        init_meg(subject)
        local_rest_raw_fname, empty_fname, cor_fname = get_meg_empty_fnames(
            subject, args.remote_meg_dir.format(subject=subject.upper()), args)
        if not op.isfile(empty_fname) or not op.isfile(cor_fname):
            print('{}: Can\'t find empty, raw, or cor files!'.format(subject))
            continue
        args = meg.read_cmd_args(dict(
            subject=subject,
            mri_subject=mri_subject,
            atlas=args.atlas,
            function='rest_functions',
            task='rest',
            reject=True, # Should be True here, unless you are dealling with bad data...
            remove_power_line_noise=True,
            l_freq=3, h_freq=80,
            windows_length=500,
            windows_shift=100,
            inverse_method='MNE',
            raw_fname=local_rest_raw_fname,
            cor_fname=cor_fname,
            empty_fname=empty_fname,
            remote_subject_dir=args.remote_subject_dir,
            # This properties are set automatically if task=='rest'
            # calc_epochs_from_raw=True,
            # single_trial_stc=True,
            # use_empty_room_for_noise_cov=True,
            # windows_num=10,
            # baseline_min=0,
            # baseline_max=0,
        ))
        meg.call_main(args)


# def calc_meg_connectivity(args):
#     args = connectivity.read_cmd_args(utils.Bag(
#         subject=args.subject,
#         atlas='laus125',
#         function='calc_lables_connectivity',
#         connectivity_modality='meg',
#         connectivity_method='pli',
#         windows_num=1,
#         # windows_length=500,
#         # windows_shift=100,
#         recalc_connectivity=True,
#         n_jobs=args.n_jobs
#     ))
#     connectivity.call_main(args)


def calc_meg_connectivity(args):
    inv_method, em = 'dSPM', 'mean_flip'
    subjects = args.subject
    for subject, mri_subject in zip(subjects, args.mri_subject):
        init_meg(subject)
        local_rest_raw_fname, empty_fname, cor_fname = get_meg_empty_fnames(
            subject, op.join(args.remote_meg_dir, subject), args) # subject.upper()
        # if not op.isfile(empty_fname) or not op.isfile(cor_fname):
        #     print('{}: Can\'t find empty, raw, or cor files!'.format(subject))
        #     continue

        # output_fname = op.join(
        #     MMVT_DIR, subject, 'connectivity', '{}_{}_coh_cwt_morlet.npz'.format(task.lower(), em))
        # if op.isfile(output_fname):
        #     file_mod_time = utils.file_modification_time_struct(output_fname)
        #     if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 11 and file_mod_time.tm_mday >= 6) or \
        #             (file_mod_time.tm_mon > 11):
        #         print('{} already exist!'.format(output_fname))
        #         continue

        remote_epo_fname = op.join(args.remote_epochs_dir, subject, args.epo_template.format(subject=subject))
        con_args = meg.read_cmd_args(utils.Bag(
            subject=subject, mri_subject=subject,
            task='rest', inverse_method=inv_method, extract_mode=em, atlas=args.atlas,
            # meg_dir=args.meg_dir,
            remote_subject_dir=args.remote_subject_dir,  # Needed for finding COR
            get_task_defaults=False,
            data_per_task=False,
            fname_format=args.epo_template.format(subject=subject)[:-len('-epo.fif')],
            epo_fname=remote_epo_fname,
            raw_fname=local_rest_raw_fname,
            empty_fname=empty_fname,
            function='make_forward_solution,calc_inverse_operator,calc_labels_connectivity',
            con_method='pli2_unbiased',
            con_mode='multitaper',
            conditions='rest',
            max_epochs_num=args.max_epochs_num,
            recreate_src_spacing='oct6p',
            check_for_channels_inconsistency=False,
            overwrite_connectivity=True,#args.overwrite_connectivity,
            cor_fname=cor_fname,
            use_empty_room_for_noise_cov=True,
            n_jobs=args.n_jobs
        ))
        meg.call_main(con_args)


def analyze_rest_fmri(gargs):
    good_subjects = []
    for subject in gargs.mri_subject:
        remote_rest_fol = get_fMRI_rest_fol(subject, gargs.remote_fmri_dir)
        if remote_rest_fol == '':
            continue
        local_rest_fname = convert_rest_dicoms_to_mgz(subject, remote_rest_fol)
        if local_rest_fname == '':
            continue
        if not op.isfile(local_rest_fname):
            print('{}: Can\'t find {}!'.format(subject, local_rest_fname))
            continue
        args = fmri.read_cmd_args(dict(
            subject=subject,
            atlas=gargs.atlas,
            remote_subject_dir=gargs.remote_subject_dir,
            function='clean_4d_data',
            fmri_file_template=local_rest_fname,
        ))
        flags = fmri.call_main(args)
        if subject not in flags or not flags[subject]['clean_4d_data']:
            continue

        args = fmri.read_cmd_args(dict(
            subject=subject,
            atlas=gargs.atlas,
            function='analyze_4d_data',
            fmri_file_template='rest.sm6.{subject}.{hemi}.mgz',
            labels_extract_mode='mean',
            overwrite_labels_data=False
        ))
        flags = fmri.call_main(args)
        if subject not in flags or not flags[subject]['analyze_4d_data']:
            continue

        args = connectivity.read_cmd_args(dict(
            subject=subject,
            atlas=gargs.atlas,
            function='calc_lables_connectivity',
            connectivity_modality='fmri',
            connectivity_method='corr',
            labels_extract_mode='mean',
            windows_length=34,  # tr = 3 -> 100s
            windows_shift=4,  # 12s
            save_mmvt_connectivity=True,
            calc_subs_connectivity=False,
            recalc_connectivity=True,
            n_jobs=gargs.n_jobs
        ))
        flags = connectivity.call_main(args)
        if subject in flags and flags[subject]['calc_lables_connectivity']:
            good_subjects.append(subject)

    print('{}/{} good subjects'.format(len(good_subjects), len(gargs.mri_subject)))
    print('Good subject: ', good_subjects)
    print('Bad subjects: ', set(gargs.mri_subject) - set(good_subjects))


def merge_connectivity(args):
    for subject in args.mri_subject:
        conn_args = connectivity.read_cmd_args(dict(subject=subject, atlas=args.atlas, norm_by_percentile=False))
        meg_con = np.abs(np.load(op.join(MMVT_DIR, subject, 'connectivity', 'meg_static_pli.npy')).squeeze())
        fmri_con = np.abs(np.load(op.join(MMVT_DIR, subject, 'connectivity', 'fmri_static_corr.npy')).squeeze())
        d = utils.Bag(np.load(op.join(MMVT_DIR, subject, 'connectivity', 'meg_static_pli.npz')))
        labels_names = np.load(op.join(MMVT_DIR, subject, 'connectivity', 'labels_names.npy'))
        meg_threshold, fmri_threshold = 0.3, 0.5
        # if args.top_k == 0:
        #     L = len(d.labels)
        #     args.top_k = int(np.rint(L * (L - 1) / 200))
        # meg_con_sparse, meg_top_k = calc_con(meg_con, args.top_k)
        # fmri_con_sparse, fmri_top_k = calc_con(fmri_con, args.top_k)

        # if len(set(fmri_top_k).intersection(set(meg_top_k))):
        #     print('fmri and meg top k intersection!')
        # con = con_fmri - con_meg
        # if len(np.where(con)[0]) != args.top_k * 2:
        #     print('Wrong number of values in the conn matrix!'.format(len(np.where(con)[0])))
        #     continue
        meg_hub, fmri_hub = calc_hubs(meg_con, fmri_con, labels_names, meg_threshold, fmri_threshold)
        meg_con_hubs, fmri_con_hubs, join_con_hubs = create_con_with_only_hubs(
            meg_con, fmri_con, meg_hub, fmri_hub, meg_threshold, fmri_threshold)
        for con_hubs, con_name in zip([meg_con_hubs, fmri_con_hubs, join_con_hubs], ['meg-hubs', 'fmri-hubs', 'fmri-meg-hubs']):
            output_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}.npz'.format(con_name))
            con_vertices_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_vertices.pkl'.format(con_name))
            connectivity.save_connectivity(
                subject, con_hubs, con_name, connectivity.ROIS_TYPE, labels_names, d.conditions, output_fname, conn_args,
                con_vertices_fname)
            print('{} was saved in {}'.format(con_name, output_fname))


def calc_con(con, top_k):
    # con[np.triu_indices(con.shape[0])] = 0
    top_k = utils.top_n_indexes(con, top_k)
    norm_con = np.zeros(con.shape)
    for top in top_k:
        norm_con[top] = con[top]
    norm_con /= np.max(norm_con)
    return norm_con, top_k


def calc_hubs(con_meg, con_fmri, labels, meg_threshold=0, fmri_threshold=0):
    # con_meg[np.triu_indices(con_meg.shape[0])] = 0
    # con_fmri[np.triu_indices(con_fmri.shape[0])] = 0
    meg_sum = np.sum(np.abs(con_meg) > meg_threshold, 0)
    fmri_sum = np.sum(np.abs(con_fmri) > fmri_threshold, 0)
    meg_hub = np.argmax(meg_sum)
    fmri_hub = np.argmax(fmri_sum)
    # intersects = np.intersect1d(np.array(np.where(con_fmri > fmri_threshold)).ravel(),
    #                             np.array(np.where(con_meg > meg_threshold)).ravel())
    # int_con = np.zeros(con_meg.shape)
    # for inter_ind in intersects:
    #     int_con[:, inter_ind] = int_con[inter_ind, :] = con_meg[:, inter_ind] + con_fmri[:, inter_ind]
    # int_hub = np.argmax(np.sum(int_con > 0, 0))
    print('meg: {}({}) {}, fmri: {}({}) {}'.format(
        labels[meg_hub], meg_hub, np.max(meg_sum),
        labels[fmri_hub], fmri_hub, np.max(fmri_sum)))
    return meg_hub, fmri_hub


def create_con_with_only_hubs(con_meg, con_fmri, meg_hub, fmri_hub, meg_threshold=0, fmri_threshold=0):
    shp = con_meg.shape
    con_join_hubs, con_meg_hubs, con_fmri_hubs, clean_con_meg, clean_con_fmri = \
        np.zeros(shp), np.zeros(shp), np.zeros(shp), np.zeros(shp), np.zeros(shp)
    con_meg_hubs[meg_hub, :] = con_meg_hubs[:, meg_hub] = -con_meg[meg_hub]
    con_fmri_hubs[fmri_hub, :] = con_fmri_hubs[:, fmri_hub] = con_fmri[fmri_hub]

    clean_con_meg[np.where(abs(con_meg) > meg_threshold)] = con_meg[np.where(abs(con_meg) > meg_threshold)]
    clean_con_fmri[np.where(abs(con_fmri) > fmri_threshold)] = con_fmri[np.where(abs(con_fmri) > fmri_threshold)]
    con_join_hubs[meg_hub, :] = con_join_hubs[:, meg_hub] = -clean_con_meg[meg_hub]
    con_join_hubs[fmri_hub, :] = con_join_hubs[:, fmri_hub] = clean_con_fmri[fmri_hub]

    # for hub in [meg_hub, fmri_hub]:
    #     meg_inds = np.where(con_meg[hub])[0]
    #     fmri_inds = np.where(con_fmri[hub])[0]
    #     int_inds = np.intersect1d(meg_inds, fmri_inds)
    #     if len(int_inds) > 0:
    #         print('Intersected connections!')
    #         # print(int_inds)
    #     con_join_hubs[hub, :] = con_join_hubs[:, hub] = con_fmri[hub] - con_meg[hub]
    return con_meg_hubs, con_fmri_hubs, con_join_hubs


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    from src.utils import preproc_utils as pu

    remote_epochs_dir = [d for d in [
        '/autofs/space/karima_002/users/Resting/epochs', op.join(MMVT_DIR, 'meg')] if op.isdir(d)][0]
    remote_subject_dir = [op.join(d, '{subject}') for d in [
        '/autofs/space/lilli_001/users/DARPA-Recons', SUBJECTS_DIR] if op.isdir(d)][0]

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='subject name', required=False, default='')
    parser.add_argument('-a', '--atlas', required=False, default='laus125')
    parser.add_argument('-f', '--function', help='function name', required=False, default='analyze_meg')
    parser.add_argument('--top_k', required=False, default=0, type=int)
    parser.add_argument('--remote_meg_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg')
    parser.add_argument('--remote_epochs_dir', required=False,default=remote_epochs_dir)
    parser.add_argument('--remote_fmri_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/mri')
    parser.add_argument('--remote_subject_dir', required=False,
                        default='/autofs/space/lilli_001/users/DARPA-Recons/{subject}')
    parser.add_argument('--epo_template', required=False, default='{subject}_Resting_meg_Demi_ar-epo.fif')
    parser.add_argument('--max_epochs_num', help='', required=False, default=10, type=int)

    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.subject = pu.decode_subjects(args.subject, remote_subject_dir=args.remote_subject_dir)
    if args.mri_subject == '':
        args.mri_subject = args.subject
    locals()[args.function](args)
