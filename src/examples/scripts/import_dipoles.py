import bpy
import os.path as op
from mathutils import Vector


def run(mmvt):
    mu = mmvt.utils
    parent_obj = mu.create_empty_if_doesnt_exists('dipoles', mmvt.BRAIN_EMPTY_LAYER)
    dipoles_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles.pkl')
    if not op.isfile(dipoles_fname):
        print('No dipoles file!')
        return
    dipoles_dict = mu.load(dipoles_fname)
    global_dipole_ind = 0
    world_matrix = mu.get_matrix_world()
    for dipole_name, dipoles in dipoles_dict.items():
        for dipole_ind, dipole in enumerate(dipoles):
            begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            dipole_loc = Vector((x, y, z)) * world_matrix
            dipole_dir = Vector((qx, qy, qz)) * world_matrix
            dipole_obj = mu.draw_arrow(global_dipole_ind, dipole_loc, dipole_dir)
            dipole_obj.name = 'dipole_{}_{}'.format(dipole_name, dipole_ind)
            dipole_obj.parent = parent_obj
            mu.create_and_set_material(dipole_obj)
            global_dipole_ind += 1
