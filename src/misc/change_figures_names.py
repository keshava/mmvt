import os.path as op
import os
import glob

from src.utils import utils


def change_figures_names(name, fol, file_type='jpeg'):
    files = glob.glob(op.join(fol, '{}_*.{}'.format(name, file_type)))
    for fname in files:
        num = int(utils.namebase(fname).split('_')[1])
        os.rename(fname, op.join(fol, '{}_new_{}.{}'.format(name, num + 1, file_type)))
    if op.isfile(op.join(fol, '{}.{}'.format(name, file_type))):
        os.rename(op.join(fol, '{}.{}'.format(name, file_type)), op.join(fol, '{}_new_1.{}'.format(name, file_type)))


if __name__ == '__main__':
    change_figures_names('rotation', '/homes/5/npeled/space1/mmvt/hbs/figures/QT1_rot')