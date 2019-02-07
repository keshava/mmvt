import os.path as op
import os
import glob

from src.utils import utils
from src.utils import figures_utils as fu
from src.utils import movies_utils as mu


def change_figures_names(name, fol, file_type='jpeg'):
    files = glob.glob(op.join(fol, '{}_*.{}'.format(name, file_type)))
    if len(files) == 0:
        raise Exception('No files!')
    for fname in files:
        num = int(utils.namebase(fname).split('_')[1])
        os.rename(fname, op.join(fol, '{}_new_{}.{}'.format(name, num + 1, file_type)))
    if op.isfile(op.join(fol, '{}.{}'.format(name, file_type))):
        os.rename(op.join(fol, '{}.{}'.format(name, file_type)), op.join(fol, '{}_new_1.{}'.format(name, file_type)))


def add_colorbar(fol, name):
    fu.add_colorbar_to_images(fol, 1, 0.95, 'RdOrYl', cb_ticks_font_size=10, cb_title=name)


def create_movie(fol, name):
    mu.combine_images(fol, name, 10, images_prefix='rotation_new_', copy_files=True)


if __name__ == '__main__':
    fol = '/home/npeled/mmvt/hbs/figures/Trop1_rotation/'
    name = 'Troponin'
    # change_figures_names('rotation', fol)
    add_colorbar(fol, name)
    create_movie(fol, name)