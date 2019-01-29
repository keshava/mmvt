import os.path as op
import os
import glob
from src.utils import figures_utils as fu
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def add_colorbar(subject, data_max=1, data_min=0.95, colors_map='BuPu_YlOrRd', file_type='jpeg'):
    figures = glob.glob(op.join(MMVT_DIR, subject, 'figures', '**', '*.{}'.format(file_type)), recursive=True)
    for figure_fname in figures:
        fu.add_colorbar_to_image(
            figure_fname, data_max, data_min, colors_map, background_color='black',
            cb_ticks=[0.95, 1], cb_ticks_font_size=10, cb_title='p-vals')
    color_bars = glob.glob(op.join(MMVT_DIR, subject, 'figures', '**', 'BuPu_YlOrRd_colorbar.jpg'), recursive=True)
    for color_bar_fname in color_bars:
        os.remove(color_bar_fname)


if __name__ == '__main__':
    add_colorbar('hbs')