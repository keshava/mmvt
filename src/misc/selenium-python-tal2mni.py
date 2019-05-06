from src.utils import trans_utils as tu
from src.utils import preproc_utils as pu
from src.utils import utils
import csv
import os.path as op
from tqdm import tqdm

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def trans_tal_coords(roi, file_name, subjects_dir, template='colin27', overwrite=False):
    subjects = {}
    output_fol = utils.make_dir(op.join(MMVT_DIR, template, 'rois_peaks'))
    csv_fname = op.join(output_fol, '{}.csv'.format(roi))
    pkl_fname = op.join(output_fol, '{}.pkl'.format(roi))
    if op.isfile(pkl_fname) and op.isfile(csv_fname) and not overwrite:
        print('Data already exist for {}'.format(roi))
        return
    driver = tu.yale_get_driver()
    files = list(utils.find_recursive(subjects_dir, file_name))
    for fname in tqdm(files):
        lines = list(utils.csv_file_reader(fname, delimiter=' '))
        subject = utils.namebase(utils.get_parent_fol(fname, 3))
        subjects[subject] = {}
        if len(lines) == 0:
            print()
            subjects[subject]['error'] = '{} is empty!'.format(fname)
            continue
        elif len(lines) > 1:
            print('More than one line in {}!'.format(fname))
            subjects[subject] = '>1'
            continue
        tal = [int(float(v)) for v in lines[0] if utils.is_float(v)]
        subjects[subject]['tal'] = tal
        subjects[subject]['mni'] = tu.yale_tal2mni(tal, driver)
    del driver
    print(subjects)
    with open(csv_fname, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for subject, subject_data in subjects.items():
            if 'mni' in subject_data:
                csv_writer.writerow(subjects[subject]['mni'])
    utils.save(subjects, pkl_fname)


if __name__ == '__main__':
    '''
    `cd /autofs/space/lilli_004/users/DARPA-MSIT/msit/subjs`
    `find . -name "dACC.txt"`
    `find . -name "L_dlPFC.txt"`
    `find . -name "R_dlPFC.txt"`
    `find . -name "L_IFG.txt"`
    `find . -name "R_IFG.txt"`
    '''
    subjects_dir = '/autofs/space/lilli_004/users/DARPA-MSIT/msit/subjs'
    rois = ['dACC', 'L_dlPFC', 'R_dlPFC', 'L_IFG', 'R_IFG']
    files = ['{}.txt'.format(roi) for roi in rois]
    for roi, file_name in zip(rois, files):
        print(roi)
        trans_tal_coords(roi, file_name, subjects_dir, 'colin27', False)