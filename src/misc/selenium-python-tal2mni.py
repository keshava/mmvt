from src.utils import trans_utils as tu
from src.utils import utils
import os
import os.path as op

SUBJECTS_DIR = '/autofs/space/lilli_004/users/DARPA-MSIT/msit/subjs'


def trans_tal_coords(file_name):
    subjects = {}
    for fname in utils.find_recursive(SUBJECTS_DIR, file_name):
        lines = list(utils.csv_file_reader(fname, delimiter=' '))
        if len(lines) == 0:
            print('{} is empty!'.format(fname))
            subjects[subject] = 'empty'
            continue
        elif len(lines) > 1:
            print('More than one line in {}!'.format(fname))
            subjects[subject] = '>1'
            break
        xyz = [int(float(v)) for v in lines[0] if utils.is_float(v)]
        subject = utils.namebase(utils.get_parent_fol(fname, 3))
        subjects[subject] = xyz
    print(subjects)


if __name__ == '__main__':
    '''
    `cd /autofs/space/lilli_004/users/DARPA-MSIT/msit/subjs`
    `find . -name "dACC.txt"`
    `find . -name "L_dlPFC.txt"`
    `find . -name "R_dlPFC.txt"`
    `find . -name "L_IFG.txt"`
    `find . -name "R_IFG.txt"`
    '''
    trans_tal_coords('dACC.txt')