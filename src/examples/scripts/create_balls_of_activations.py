import bpy
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    csv_fname = mu.get_real_fname('balls_of_activations_csv_fname')
    balls_r = {1:1, 2:2}
    balls_c = {1:'blue', 2:'red'}
    if not op.isfile(csv_fname):
        print('Can\'t find csv fname! {}!'.format(csv_fname))
        return False
    t1_trans = mmvt.where_am_i.t1_trans()
    for ind, line in enumerate(mu.csv_file_reader(csv_fname, find_encoding=True)):
        x, y, z = [float(x) for x in line[:3]]
        cond, primary = int(line[3]), int(line[4])
        ras_tkr = mu.apply_trans(t1_trans.vox2ras_tkr, [x, y, z])
        _, _, vertex_co = mmvt.vertex_data.find_closest_vertex_index_and_mesh(ras_tkr)
        ball_name = 'peak_{}_{}_{}'.format(ind, cond, primary)
        mmvt.data.create_electrode(vertex_co, ball_name, balls_r[primary], color=balls_c[cond])


bpy.types.Scene.balls_of_activations_csv_fname = bpy.props.StringProperty(subtype='FILE_PATH')


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'balls_of_activations_csv_fname', text='csv')
