import bpy
import os.path as op
from scripts_panel import ScriptsPanel
from mathutils import Vector


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    pass


def dipoles_names_update(self, context):
    mmvt, mu = _mmvt(), _mmvt().utils
    selected_cluster_name = bpy.context.scene.dipoles_names
    dipoles = ScriptsPanel.dipoles_dict[selected_cluster_name]
    begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipoles[0]
    dipole_loc = Vector((x, y, z)) * mu.get_matrix_world() * 1e3
    mmvt.where_am_i.set_cursor_location(dipole_loc)


def load_dipoles():
    mmvt, mu = _mmvt(), _mmvt().utils
    parent_obj = mu.create_empty_if_doesnt_exists('dipoles', mmvt.MEG_LAYER, None, 'Functional maps')
    dipoles_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles.pkl')
    if not op.isfile(dipoles_fname):
        print('No dipoles file!')
        return
    dipoles_dict = ScriptsPanel.dipoles_dict #mu.load(dipoles_fname)
    world_matrix = mu.get_matrix_world()
    layers_array = [False] * 20
    layers_array[mmvt.MEG_LAYER] = True
    show_as_arrows = True
    color = (1, 0, 0, 1)
    for dipole_name, dipoles in dipoles_dict.items():
        for dipole_ind, dipole in enumerate(dipoles):
            dipole_obj_name = 'dipole_{}_{}'.format(dipole_name, dipole_ind)
            begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            dipole_loc = Vector((x, y, z)) * world_matrix * 1e3
            # print(dipole_loc)
            ori = Vector((qx, qy, qz))
            dipole_dir = ori * world_matrix * 15
            # print(dipole_dir)
            # print(dipole_loc + dipole_dir)
            # dipole_obj = draw_arrow(global_dipole_ind, dipole_loc, dipole_dir)
            mu.create_sphere(dipole_loc, 0.15, layers_array, dipole_obj_name)
            dipole_obj = bpy.data.objects[dipole_obj_name]
            direction_obj = mu.cylinder_between(
                dipole_loc, dipole_loc + dipole_dir, 0.1, layers_array, '{}_direction'.format(dipole_obj_name), color)
            for obj in [dipole_obj, direction_obj]:
                obj.select = True
                obj.parent = parent_obj
            mu.create_and_set_material(dipole_obj)
    bpy.context.scene.layers[mmvt.MEG_LAYER] = True


def delete_dipoles():
    mu = _mmvt().mmvt_utils
    mu.delete_hierarchy('dipoles')


class LoadDipoles(bpy.types.Operator):
    bl_idname = "mmvt.load_delete"
    bl_label = "mmvt load dipoles"
    bl_description = 'Load the dipoles'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        load_dipoles()
        return {"FINISHED"}


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
    mu = _mmvt().utils

    # layout.prop(context.scene, 'dipoles_show_as_arrow', text="Plot as arrow")
    layout.operator(LoadDipoles.bl_idname, text="Import Dipoles", icon='EDITMODE_HLT')
    layout.prop(context.scene, 'dipoles_names', text="")

    if ScriptsPanel.dipoles_rois is not None and bpy.context.scene.dipoles_names in ScriptsPanel.dipoles_rois:
        layout.label(text='Dipole\'s ROIs:')
        box = layout.box()
        col = box.column()
        rois = ScriptsPanel.dipoles_rois[bpy.context.scene.dipoles_names]
        for subcortical_name, subcortical_prob in zip(rois['subcortical_rois'], rois['subcortical_probs']):
            mu.add_box_line(col, subcortical_name, '{:.3f}'.format(subcortical_prob), 0.8)
        for cortical_name, cortical_prob in zip(rois['cortical_rois'], rois['cortical_probs']):
            if cortical_prob >= 0.001:
                mu.add_box_line(col, cortical_name, '{:.3f}'.format(cortical_prob), 0.8)
    layout.operator(DeleteDipoles.bl_idname, text="Delete Dipoles", icon='PANEL_CLOSE')


bpy.types.Scene.dipoles_names = bpy.props.EnumProperty(items=[], description="Dipoles names")
bpy.types.Scene.dipoles_show_as_arrow = bpy.props.BoolProperty()


def init(mmvt):
    mu = mmvt.mmvt_utils
    dipoles_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles.pkl')
    if not op.isfile(dipoles_fname):
        print('No dipoles file!')
        return
    ScriptsPanel.dipoles_dict = dipoles_dict = mu.load(dipoles_fname)
    dipoles_items = sorted([(dipole_name, dipole_name, '', ind) for ind, dipole_name in enumerate(dipoles_dict.keys())])
    bpy.types.Scene.dipoles_names = bpy.props.EnumProperty(
        items=dipoles_items, description="Dipoles names", update=dipoles_names_update)
    dipoles_rois_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles_rois.pkl')
    if op.isfile(dipoles_rois_fname):
        ScriptsPanel.dipoles_rois = mu.load(dipoles_rois_fname)
    else:
        ScriptsPanel.dipoles_rois = None
    register()


def register():
    try:
        bpy.utils.register_class(LoadDipoles)
        bpy.utils.register_class(DeleteDipoles)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(LoadDipoles)
        bpy.utils.unregister_class(DeleteDipoles)
    except:
        pass



#
