import bpy
import os.path as op
from scripts_panel import ScriptsPanel
from mathutils import Vector


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    mu = mmvt.utils
    parent_obj = mu.create_empty_if_doesnt_exists('dipoles', mmvt.MEG_LAYER, None, 'Functional maps')
    dipoles_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles.pkl')
    if not op.isfile(dipoles_fname):
        print('No dipoles file!')
        return
    dipoles_dict = mu.load(dipoles_fname)
    global_dipole_ind = 0
    world_matrix = mu.get_matrix_world()
    layers_array = [False] * 20
    layers_array[mmvt.MEG_LAYER] = True
    show_as_arrows = False
    for dipole_name, dipoles in dipoles_dict.items():
        for dipole_ind, dipole in enumerate(dipoles):
            dipole_obj_name = 'dipole_{}_{}'.format(dipole_name, dipole_ind)
            begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            dipole_loc = Vector((x, y, z)) * world_matrix * 1000
            ori = Vector((qx, qy, qz)) / (1 / 1e9)
            dipole_dir = ori * world_matrix * 1000
            if show_as_arrows:
                dipole_obj = mu.draw_arrow(global_dipole_ind, dipole_loc, dipole_dir)
            else:
                mu.create_sphere(dipole_loc, 0.15, layers_array, dipole_obj_name)
                dipole_obj = bpy.data.objects[dipole_obj_name]
            dipole_obj.select = True
            dipole_obj.parent = parent_obj
            mu.create_and_set_material(dipole_obj)
            global_dipole_ind += 1
    bpy.context.scene.layers[mmvt.MEG_LAYER] = True

def delete_dipoles():
    mu = _mmvt().mmvt_utils
    mu.delete_hierarchy('dipoles')


class DeleteDipoles(bpy.types.Operator):
    bl_idname = "mmvt.dipoles_delete"
    bl_label = "mmvt dipoles delete"
    bl_description = 'Delete the dipoles'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        delete_dipoles()
        return {"FINISHED"}


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'dipoles_names', text="")
    layout.operator(DeleteDipoles.bl_idname, text="Delete Dipoles", icon='FORCE_HARMONIC')


bpy.types.Scene.dipoles_names = bpy.props.EnumProperty(items=[], description="Dipoles names")


def init(mmvt):
    mu = mmvt.mmvt_utils
    dipoles_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles.pkl')
    if not op.isfile(dipoles_fname):
        print('No dipoles file!')
        return
    dipoles_dict = mu.load(dipoles_fname)
    dipoles_items = sorted([(dipole_name, dipole_name, '', ind) for ind, dipole_name in enumerate(dipoles_dict.keys())])
    bpy.types.Scene.dipoles_names = bpy.props.EnumProperty(
        items=dipoles_items, description="Dipoles names")
    register()



def register():
    try:
        bpy.utils.register_class(DeleteDipoles)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(DeleteDipoles)
    except:
        pass



