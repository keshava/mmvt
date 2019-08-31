import os.path as op
import nibabel as nib
import numpy as np
import traceback
from collections import defaultdict
from src.utils import utils
from src.utils import preproc_utils as pu
# from src.examples import morph_electrodes_to_template
from src.preproc import anatomy as anat, ela_morph_electrodes

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def read_xls(xls_fname,  specific_subjects=None):
    subjects_electrodes = defaultdict(list)
    for line in utils.xlsx_reader(xls_fname, skip_rows=1):
        subject, elec1_name, elec2_name, cond, patient_id  = line[:5]
        subject = subject.lower()
        if specific_subjects is not None and subject not in specific_subjects:
            continue
        elec1_coo, elec2_coo = line[5:8], line[8:11]
        subjects_electrodes[subject].append(elec1_name)
        subjects_electrodes[subject].append(elec2_name)
    return subjects_electrodes


def morph_electrodes(subjects_electrodes, subject_to='colin27', atlas='aparc.DKTatlas', annotation_template='fsaverage',
                     overwrite=False, n_jobs=1):
    subjects = list(subjects_electrodes.keys())
    indices = np.array_split(np.arange(len(subjects)), n_jobs)
    chunks = [([subjects[ind] for ind in chunk_indices], atlas, subject_to, subjects_electrodes, annotation_template,
               overwrite) for chunk_indices in indices]
    results = utils.run_parallel(_morph_electrodes_parallel, chunks, n_jobs)
    for bad_subjects in results:
        for bad_subject in bad_subjects:
            print(bad_subject)


def _morph_electrodes_parallel(p):
    subjects, atlas, subject_to, subjects_electrodes, annotation_template, overwrite = p
    bad_subjects = []
    for subject in subjects:
        get_subject_files_from_mad([subject], atlas)
        atlas = utils.fix_atlas_name(subject, atlas, SUBJECTS_DIR)
        if not utils.both_hemi_files_exist(
                op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
            err = ''
            try:
                anat.create_annotation(subject, atlas, annotation_template, n_jobs=1, overwrite_vertices_labels_lookup=True)
            except:
                print(traceback.format_exc())
                err = utils.print_last_error_line()
            if not utils.both_hemi_files_exist(
                    op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
                bad_subjects.append((subject, 'No atlas' if err == '' else err))
                continue
        try:
            subjects_electrodes[subject] = list(set(subjects_electrodes[subject]))
            if not overwrite:
                subjects_electrodes[subject] = [elc_name for elc_name in subjects_electrodes[subject] if not op.isfile(
                    op.join(MMVT_DIR, subject, 'electrodes', '{}_ela_morphed.npz'.format(elc_name)))]
            ela_morph_electrodes.calc_elas(
                subject, subject_to, subjects_electrodes[subject], bipolar=False, atlas=atlas, overwrite=overwrite,
                n_jobs=1)
        except:
            print(traceback.format_exc())
            err = utils.print_last_error_line()
            bad_subjects.append((subject, err))
    return bad_subjects


def read_morphed_electrodes(subjects_electrodes, subject_to='colin27', bipolar=True, prefix='morphed_', postfix=''):
    output_fname = '{}electrodes{}_positions.npz{}'.format(prefix, '_bipolar' if bipolar else '', postfix)
    bad_electrodes = []
    template_electrodes = defaultdict(list)
    morphed_electrodes_fname = op.join(MMVT_DIR, subject_to, 'electrodes', 'morphed_electrodes.pkl')
    if op.isfile(morphed_electrodes_fname):
        template_electrodes = utils.load(morphed_electrodes_fname)
    else:
        elecs_pos = []
        for subject, electodes_names in subjects_electrodes.items():
            for elec_name in electodes_names:
                elec_input_fname = op.join(MMVT_DIR, subject, 'electrodes', 'ela_morphed',
                                           '{}_ela_morphed.npz'.format(elec_name))
                if not op.isfile(elec_input_fname):
                    print('{} {} not found!'.format(subject, elec_name))
                    bad_electrodes.append('{}_{}'.format(subject, elec_name))
                else:
                    d = np.load(elec_input_fname)
                    template_electrodes[subject].append((elec_name, d['pos']))
                    elecs_pos.append(d['pos'])
        utils.save(template_electrodes, morphed_electrodes_fname)

    fol = utils.make_dir(op.join(MMVT_DIR, subject_to, 'electrodes'))
    output_fname = op.join(fol, output_fname)
    elecs_coordinates = np.array(utils.flat_list_of_lists(
        [[e[1] for e in template_electrodes[subject]] for subject in template_electrodes.keys()]))
    elecs_names = utils.flat_list_of_lists(
        [['{}_{}'.format(subject.upper(), e[0]) for e in template_electrodes[subject]] for subject in
         template_electrodes.keys()])
    print('Saving {} electrodes in {}:'.format(subject_to, output_fname))
    # print(elecs_names)
    np.savez(output_fname, pos=elecs_coordinates, names=elecs_names, pos_org=[])

    print('Bad electrodes:')
    print(bad_electrodes)

    return template_electrodes


def write_electrode_colors(template, electrodes_colors):
    import csv
    fol = utils.make_dir(op.join(MMVT_DIR, template, 'coloring'))
    csv_fname = op.join(fol, 'morphed_electrodes.csv')
    # unique_colors = np.unique(utils.flat_list(([[k[1] for k in elecs] for elecs in electrodes_colors.values()])))
    # colors = utils.get_distinct_colors(len(unique_colors))
    from src.mmvt_addon import colors_utils as cu
    colors = [cu.name_to_rgb(c) for c in ['blue', 'green', 'purple', 'red', 'brown', 'yellow']]
    print('Writing csv file to {}'.format(csv_fname))
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
        for subject in electrodes_colors.keys():
            # for elc_name, _ in template_electrodes[subject]:
            for elc_name, color_id in electrodes_colors[subject]:
                color = colors[color_id - 1]
                wr.writerow([elc_name, *color])


def get_subject_files_from_mad(subjects, atlas):
    for subject in subjects:
        root_fol = '/mnt/cashlab/Original Data/{}'.format(subject[:2].upper())
        args = anat.read_cmd_args(dict(
            subject=subject,
            atlas=atlas,
            remote_subject_dir=op.join(root_fol, '{subject}/{subject}_Notes_and_Images/{subject}_SurferOutput'),
            function='prepare_subject_folder',
            ignore_missing=1,
        ))
        pu.run_on_subjects(args, anat.main)


if __name__ == '__main__':
    fols = ['/home/npeled/Documents/Angelique/mapping_to_common_brains',
            '/autofs/space/thibault_001/users/npeled/Documents/Angelique/mapping_to_common_brains']
    fol = [f for f in fols if op.isdir(f)][0]
    xls_fname = op.join(fol, 'ChannelListFull.xls')
    bipolar = True
    subject_to = 'colin27'
    atlas = 'laus125'
    annotation_template = 'fsaverage5c'
    overwrite = False
    n_jobs = 10
    specific_subjects = None #['mg78']

    subjects_electrodes = read_xls(xls_fname, specific_subjects)
    # morph_electrodes(subjects_electrodes, subject_to, atlas, annotation_template, overwrite, n_jobs)
    subjects_electrodes, electrodes_colors = read_morphed_electrodes(subjects_electrodes, subject_to)
    # morph_electrodes_to_template.export_into_csv(subjects_electrodes, template_system, MMVT_DIR, bipolar)
    # morph_electrodes_to_template.create_mmvt_coloring_file(template_system, subjects_electrodes, electrodes_colors)
