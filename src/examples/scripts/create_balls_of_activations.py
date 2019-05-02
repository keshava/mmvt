import bpy
import os.path as op
import numpy as np
import time


def run(mmvt):
    mu = mmvt.utils
    csv_fname = mu.get_real_fname('balls_of_activations_csv_fname')
    balls_r = {1: 0.1, 2: 0.2}
    balls_c = {1: 'blue', 2: 'red'}
    if not op.isfile(csv_fname):
        print('Can\'t find csv fname! {}!'.format(csv_fname))
        return False
    lines = list(mu.csv_file_reader(csv_fname, find_encoding=True))
    now = time.time()
    for ind, line in enumerate(lines):
        mu.time_to_go(now, ind, len(lines), runs_num_to_print=10)
        mni305_ras = np.array([float(x) for x in line[:3]])
        cond, primary = int(line[3]), int(line[4])
        subject_tkreg_ras = mmvt.where_am_i.mni305_ras_to_subject_tkreg_ras(mni305_ras)
        inflated_pos = mmvt.where_am_i.pos_to_current_inflation(subject_tkreg_ras, subject_tkreg_ras=True)
        ball_name = 'peak_{}_{}_{}'.format(ind, cond, primary)
        mmvt.data.create_electrode(
            inflated_pos, ball_name, balls_r[primary], color=balls_c[cond], subject_tkreg_ras=True)
        mmvt.appearance.show_electrodes(True)


bpy.types.Scene.balls_of_activations_csv_fname = bpy.props.StringProperty(subtype='FILE_PATH')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'balls_of_activations_csv_fname', text='csv')
