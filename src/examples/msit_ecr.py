import os.path as op
from src.utils import utils
from src.preproc import anatomy as anat
from src.preproc import meg as meg
from src.preproc import connectivity as con


LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def anatomy_preproc(args):
    args = anat.read_cmd_args(dict(
        subject=args.subject,
        remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}',
        high_level_atlas_name='darpa_atlas'
    ))
    anat.call_main(args)


def init_meg_preproc(args, tasks):
    for subject in args.subject:
        utils.make_dir(op.join(MEG_DIR, subject))
        utils.make_link(op.join(args.remote_subject_dir.format(subject=subject), 'bem'),
                        op.join(MEG_DIR, subject, 'bem'))
        for task in tasks:
            utils.make_dir(op.join(MEG_DIR, task, subject))
            utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(MEG_DIR, task, subject, 'bem'))
        utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'))

        remote_meg_fol = '/autofs/space/lilli_003/users/DARPA-TRANSFER/meg/{}'.format(subject)
        csv_fname = op.join(remote_meg_fol, 'cfg.txt')
        days, empty_fnames = {}, {}
        for line in utils.csv_file_reader(csv_fname, ' '):
            for task in tasks:
                if line[4].lower() == task.lower():
                    days[task] = line[2]
        print(days)
        for line in utils.csv_file_reader(csv_fname, ' '):
            if line[4] == 'empty':
                for task in tasks:
                    empty_fnames[task] = op.join(MEG_DIR, task, subject, '{}_empty_raw.fif'.format(subject))
                    if op.isfile(empty_fnames[task]):
                        continue
                    task_day = days[task]
                    if line[2] == task_day:
                        empty_fname = op.join(remote_meg_fol, line[0].zfill(3), line[-1])
                        if not op.isfile(empty_fname):
                            raise Exception('empty file does not exist! {}'.format(empty_fname[task]))
                        utils.make_link(empty_fname, empty_fnames[task])
        return empty_fnames


def meg_preproc(args):
    atlas, inv_method, em = 'aparc.DKTatlas40', 'dSPM', 'mean_flip'
    atlas = 'darpa_atlas'
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    tasks = ['MSIT', 'ECR']
    empty_fnames = init_meg_preproc(args, tasks)
    times = (-2, 4)
    for task in tasks:
        args = meg.read_cmd_args(dict(
            subject=args.subject, mri_subject=args.subject,
            task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
            remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}', # Needed for finding COR
            get_task_defaults=False,
            fname_format='{}_{}_nTSSS-ica-raw'.format('{subject}', task.lower()),
            empty_fname=empty_fnames[task],
            # function='calc_epochs,calc_evokes,make_forward_solution,calc_inverse_operator,calc_stc_per_condition,calc_labels_avg_per_condition,calc_labels_min_max',
            function='calc_epochs,calc_evokes,make_forward_solution,calc_inverse_operator',
            # function='calc_stc_per_condition',
            # function='calc_labels_connectivity',
            conditions=task.lower(),
            data_per_task=True,
            ica_overwrite_raw=False,
            normalize_data=False,
            t_min=times[0], t_max=times[1],
            read_events_from_file=False, stim_channels='STI001',
            use_empty_room_for_noise_cov=True,
            calc_source_band_induced_power=True,
            calc_inducde_power_per_label=True,
            bands='', #dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200]),
            con_method='coh',
            con_mode='cwt_morlet',
            overwrite_connectivity=False,
            read_only_from_annot=False,
            # pick_ori='normal',
            # overwrite_epochs=True,
            # overwrite_evoked=True,
            overwrite_inv=True,
            overwrite_stc=True,
            overwrite_labels_data=True,
            n_jobs=args.n_jobs
        ))
        meg.call_main(args)
    #
    for subject in args.subject:
        for task in tasks:
            task = task.lower()
            # meg.calc_labels_func(subject, task, atlas, em, tmin=0, tmax=0.5, times=times, norm_data=False)
            # meg.calc_labels_power_bands(subject, task, atlas, em, tmin=times[0], tmax=times[1], overwrite=True)

    #     meg_dir = op.join(MMVT_DIR, subject, 'meg')
    #     # meg.calc_stc_diff(op.join(meg_dir, '{}_msit-{}-rh.stc'.format(subject, inv_method)),
    #     #                   op.join(meg_dir, '{}_ecr-{}-rh.stc'.format(subject, inv_method)),
    #     #                   op.join(meg_dir, '{}-msit-ecr-{}-rh.stc'.format(subject, inv_method)))
    #     meg.calc_labels_diff(
    #         op.join(meg_dir, 'labels_data_msit_{}_{}_{}.npz'.format(atlas, em, '{hemi}')),
    #         op.join(meg_dir, 'labels_data_ecr_{}_{}_{}.npz'.format(atlas, em, '{hemi}')),
    #         op.join(meg_dir, 'labels_data_msit-ecr_{}_{}_{}.npz'.format(atlas, em, '{hemi}')), norm_data=True)
    #


# def con_preprc(args):
#     atlas = 'aparc.DKTatlas40'
#     for task in ['MSIT', 'ECR']:
#         args = con.read_cmd_args(utils.Bag(
#             subject=args.subject,
#             atlas=atlas,
#             function='calc_lables_connectivity',
#             labels_data_name='labels_data_{}_{}_mean_flip_{}.npz'.format(task.lower(), atlas, '{hemi}'),
#             connectivity_modality='meg',
#             connectivity_method='coherence',
#             identifier=task.lower(),
#             windows_length=6000,
#             windows_shift=100,
#             sfreq=1000.0,
#             # fmin=10,
#             # fmax=100
#             # recalc_connectivity=True,
#             max_windows_num=1,
#             recalc_connectivity=True,
#             n_jobs=args.n_jobs
#         ))
#         con.call_main(args)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.remote_subject_dir = '/autofs/space/lilli_001/users/DARPA-Recons/{subject}'
    locals()[args.function](args)