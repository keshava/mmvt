import bpy
import os.path as op
import glob
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
    # begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipoles[0]

    dipoles_list_of_tuples = sorted([(str(item[0]), str(item[0]), '', ind) for ind, item in enumerate(dipoles)])
    bpy.types.Scene.dipoles_sub_group = bpy.props.EnumProperty(
        items=dipoles_list_of_tuples, description="Dipoles sub group",
        update=dipoles_sub_group_update)
    bpy.context.scene.dipoles_sub_group = dipoles_list_of_tuples[0][0]


def dipoles_sub_group_update(self, context):
    mmvt, mu = _mmvt(), _mmvt().utils
    selected_cluster_name = bpy.context.scene.dipoles_names
    selected_dipole_name = bpy.context.scene.dipoles_sub_group
    dipole = ScriptsPanel.dipoles_dict[selected_cluster_name][0]
    for dip in ScriptsPanel.dipoles_dict[selected_cluster_name]:
        if float(selected_dipole_name) == dip[0]:
            dipole = dip
            break
    dipole_loc = Vector((dipole[2], dipole[3], dipole[4])) * mu.get_matrix_world() * 1e3
    mmvt.where_am_i.set_cursor_location(dipole_loc)


def dipoles_color_update(self, context):
    mmvt, mu = _mmvt(), _mmvt().utils
    for dipole_name, dipoles in ScriptsPanel.dipoles_dict.items():
        for dipole in dipoles:
            dipole_obj_name = 'dipole_{}_{}_direction'.format(dipole_name, dipole[0])
            mmvt.coloring.object_coloring(dipole_obj_name, bpy.context.scene.dipoles_color)


def dipoles_connectivity_update(self, context):
    mmvt, mu = _mmvt(), _mmvt().utils
    connectivity_name = bpy.context.scene.dipoles_connectivity
    connectivity_fnames = [f for f in ScriptsPanel.connections_files if mu.namebase(f).startswith(connectivity_name)]
    if len(connectivity_fnames) == 0:
        print('dipoles_connectivity_update: the connectivity file could not be found!')
        return
    ScriptsPanel.dipole_connections = connections = mu.load(connectivity_fnames[0])
    ScriptsPanel.dipole_connections_item_names = [
        '{}) {:.4f}ms'.format(ind + 1, info[0]) for ind, info in enumerate(connections)]
    dipoles_connections_items = \
        sorted([(item, item , '', ind) for ind, item in enumerate(ScriptsPanel.dipole_connections_item_names)])
    bpy.types.Scene.dipole_connections = bpy.props.EnumProperty(
        items=dipoles_connections_items, description="Dipoles connections",
        update=dipole_connections_update)
    bpy.context.scene.dipole_connections = ScriptsPanel.dipole_connections_item_names[0]


def dipole_connections_update(self, context):
    dipole_current_connection = bpy.context.scene.dipole_connections
    dipole_current_connection_id = ScriptsPanel.dipole_connections_item_names.index(dipole_current_connection)# int(dipole_current_connection.split(')')[0]) - 1
    ScriptsPanel.dipole_connection = ScriptsPanel.dipole_connections[dipole_current_connection_id]
    if ScriptsPanel.labels_pos is not None:
        con = ScriptsPanel.dipole_connection[-1].split(' ')[0]
        name_parts = con.split('-')
        from_label_name = '{}_{}-{}'.format(name_parts[0], name_parts[1], name_parts[3])
        to_label_name = '{}_{}-{}'.format(name_parts[4], name_parts[5], name_parts[7])
        from_pos = ScriptsPanel.labels_pos[from_label_name] * 100
        to_pos = ScriptsPanel.labels_pos[to_label_name] * 100
        update_connection_obj(from_pos, to_pos)
    else:
        print('labels_pos is None')


def update_connection_obj(from_pos, to_pos, color=(0, 1, 0, 1)):
    # todo: check if the cylender info can be updated
    # todo: use a cone to show the direction
    # https://blender.stackexchange.com/questions/31405/blender-orient-an-object-displaying-vector-arrows
    # todo: what to do if from_pos = to_pos?
    mmvt, mu = _mmvt(), _mmvt().utils
    parent_obj = mu.create_empty_if_doesnt_exists('dipoles', mmvt.MEG_LAYER, None, 'Functional maps')
    layers_array = [False] * 20
    layers_array[mmvt.MEG_LAYER] = True

    con_obj_name = 'dipoles_con'
    mu.delete_object(con_obj_name)
    con_obj = mu.cylinder_between(
        from_pos, to_pos, 0.1, layers_array, con_obj_name, color)
    con_obj.select = True
    con_obj.parent = parent_obj
    mu.create_and_set_material(con_obj)


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
        for dipole in dipoles:
            begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            dipole_obj_name = 'dipole_{}_{}'.format(dipole_name, begin_t)
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


def get_connection_type_name(con_type):
    if con_type == 'between to rh':
        return 'left -> right'
    elif con_type == 'between to lh':
        return 'right -> left'
    elif con_type == 'within rh':
        return 'right -> right'
    elif con_type == 'within lh':
        return 'left -> left'


def next_dipole_connection():
    values = [k[0] for k in bpy.types.Scene.dipole_connections[1]['items']]
    index = values.index(bpy.context.scene.dipole_connections)
    bpy.context.scene.dipole_connections = values[index + 1] if index < len(values) - 1 else values[0]


def prev_dipole_connection():
    values = [k[0] for k in bpy.types.Scene.dipole_connections[1]['items']]
    index = values.index(bpy.context.scene.dipole_connections)
    bpy.context.scene.dipole_connections = values[index - 1] if index > 0 else values[-1]


def next_dipole():
    values = [k[0] for k in bpy.types.Scene.dipoles_names[1]['items']]
    index = values.index(bpy.context.scene.dipoles_names)
    bpy.context.scene.dipoles_names = values[index + 1] if index < len(values) - 1 else values[0]


def prev_dipole():
    values = [k[0] for k in bpy.types.Scene.dipoles_names[1]['items']]
    index = values.index(bpy.context.scene.dipoles_names)
    bpy.context.scene.dipoles_names = values[index - 1] if index > 0 else values[-1]


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


class NextDipoleConnection(bpy.types.Operator):
    bl_idname = 'mmvt.next_dipole_connection'
    bl_label = 'next dipole_connection'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_dipole_connection()
        return {'FINISHED'}


class PrevDipoleConnection(bpy.types.Operator):
    bl_idname = 'mmvt.prev_dipole_connection'
    bl_label = 'prev dipole_connection'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_dipole_connection()
        return {'FINISHED'}


class NextDipole(bpy.types.Operator):
    bl_idname = 'mmvt.next_dipole'
    bl_label = 'next dipole'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        next_dipole()
        return {'FINISHED'}


class PrevDipole(bpy.types.Operator):
    bl_idname = 'mmvt.prev_dipole'
    bl_label = 'prev dipole'
    bl_options = {'UNDO'}

    def invoke(self, context, event=None):
        prev_dipole()
        return {'FINISHED'}


def draw(self, context):
    layout = self.layout
    mu = _mmvt().utils

    # layout.prop(context.scene, 'dipoles_show_as_arrow', text="Plot as arrow")
    layout.operator(LoadDipoles.bl_idname, text="Import Dipoles", icon='EDITMODE_HLT')
    row = layout.row(align=True)
    row.operator(PrevDipole.bl_idname, text="", icon='PREV_KEYFRAME')
    row.prop(context.scene, 'dipoles_names', text="")
    row.operator(NextDipole.bl_idname, text="", icon='NEXT_KEYFRAME')
    first_dipole = '{}_{}'.format(bpy.context.scene.dipoles_names, ScriptsPanel.dipoles_dict[
        bpy.context.scene.dipoles_names][0][0])
    layout.prop(context.scene, 'dipoles_color', text="")
    if len(ScriptsPanel.dipoles_dict[bpy.context.scene.dipoles_names]) > 1:
        layout.prop(context.scene, 'dipoles_sub_group', text="")
    if ScriptsPanel.dipoles_rois is not None and bpy.context.scene.dipoles_names in ScriptsPanel.dipoles_rois or \
            first_dipole in ScriptsPanel.dipoles_rois:
        layout.label(text='Dipole\'s ROIs:')
        box = layout.box()
        col = box.column()
        if bpy.context.scene.dipoles_names in ScriptsPanel.dipoles_rois:
            rois = ScriptsPanel.dipoles_rois[bpy.context.scene.dipoles_names]
        else:
            rois = ScriptsPanel.dipoles_rois[first_dipole]
        for subcortical_name, subcortical_prob in zip(rois['subcortical_rois'], rois['subcortical_probs']):
            mu.add_box_line(col, subcortical_name, '{:.3f}'.format(subcortical_prob), 0.8)
        for cortical_name, cortical_prob in zip(rois['cortical_rois'], rois['cortical_probs']):
            # if cortical_prob >= 0.001:
            mu.add_box_line(col, cortical_name, '{:.3f}'.format(cortical_prob), 0.8)

    if False: #len(ScriptsPanel.connections_files) > 0:
        layout.label(text='Dipole\'s connections:')
        layout.prop(context.scene, 'dipoles_connectivity', text="")

        if ScriptsPanel.dipole_connection is not None:
            row = layout.row(align=True)
            row.operator(PrevDipoleConnection.bl_idname, text="", icon='PREV_KEYFRAME')
            row.prop(context.scene, 'dipole_connections', text="")
            row.operator(NextDipoleConnection.bl_idname, text="", icon='NEXT_KEYFRAME')
            box = layout.box()
            col = box.column()
            con = ScriptsPanel.dipole_connection
            con_name = con[3].split(' ')[0]
            from_label, _, _, from_hemi, to_label, _, _ , to_hemi = con_name.split('-')
            col.label(text='onset: {:.4f}ms'.format(con[0]))
            col.label(text=get_connection_type_name(con[1]))
            col.label(text='from: {}-{}'.format(from_label, from_hemi))
            col.label(text='to: {}-{}'.format(to_label, to_hemi))
            col.label(text='strength: {:.2f}'.format(con[2]))

    layout.operator(DeleteDipoles.bl_idname, text="Delete Dipoles", icon='PANEL_CLOSE')


bpy.types.Scene.dipoles_names = bpy.props.EnumProperty(items=[], description="Dipoles names")
bpy.types.Scene.dipoles_connectivity = bpy.props.EnumProperty(items=[], description="Dipoles connectivity")
bpy.types.Scene.dipole_connections = bpy.props.EnumProperty(items=[], description="Dipole connections")
bpy.types.Scene.dipoles_show_as_arrow = bpy.props.BoolProperty()
bpy.types.Scene.dipoles_sub_group = bpy.props.EnumProperty(items=[], description="Dipoles sub group")
bpy.types.Scene.dipoles_color = bpy.props.FloatVectorProperty(
    name="object_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0,
    description='Selects the color to color the dipoles', update=dipoles_color_update)

def init(mmvt):
    mu = mmvt.mmvt_utils
    ScriptsPanel.dipoles_dict = {}
    ScriptsPanel.connections_files = []
    ScriptsPanel.dipole_connection = None
    ScriptsPanel.labels_pos = None

    dipoles_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles.pkl')
    if not op.isfile(dipoles_fname):
        print('No dipoles file!')
        return
    ScriptsPanel.dipoles_dict = dipoles_dict = mu.load(dipoles_fname)
    dipoles_names = sorted(list(dipoles_dict.keys()))
    dipoles_items = [(dipole_name, dipole_name, '', ind) for ind, dipole_name in enumerate(dipoles_names)]
    bpy.types.Scene.dipoles_names = bpy.props.EnumProperty(
        items=dipoles_items, description="Dipoles names", update=dipoles_names_update)
    bpy.context.scene.dipoles_names = dipoles_names[0]

    dipoles_rois_fname = op.join(mu.get_user_fol(), 'meg', 'dipoles_rois.pkl')
    if op.isfile(dipoles_rois_fname):
        ScriptsPanel.dipoles_rois = mu.load(dipoles_rois_fname)
    else:
        ScriptsPanel.dipoles_rois = None

    # todo: load the altas name
    labels_pos_fname = op.join(mu.get_user_fol(), 'laus125_center_of_mass.pkl')
    if op.isfile(labels_pos_fname):
        ScriptsPanel.labels_pos = mu.load(labels_pos_fname)

    ScriptsPanel.connections_files = glob.glob(op.join(mu.get_user_fol(), 'connectivity', 'dipoles', '*.pkl'))
    if len(ScriptsPanel.connections_files) > 0:
        dipoles_connections_names = [mu.namebase(f).split('-')[0] for f in ScriptsPanel.connections_files]
        dipoles_connections_items = sorted([(n, n , '', ind) for ind, n in enumerate(dipoles_connections_names)])
        bpy.types.Scene.dipoles_connectivity = bpy.props.EnumProperty(
            items=dipoles_connections_items, description="Dipoles connections",
            update=dipoles_connectivity_update)
        bpy.context.scene.dipoles_connectivity = dipoles_connections_items[0][0]
    register()


def register():
    try:
        bpy.utils.register_class(LoadDipoles)
        bpy.utils.register_class(DeleteDipoles)
        bpy.utils.register_class(NextDipole)
        bpy.utils.register_class(PrevDipole)
        bpy.utils.register_class(NextDipoleConnection)
        bpy.utils.register_class(PrevDipoleConnection)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(LoadDipoles)
        bpy.utils.unregister_class(DeleteDipoles)
        bpy.utils.unregister_class(NextDipole)
        bpy.utils.unregister_class(PrevDipole)
        bpy.utils.unregister_class(NextDipoleConnection)
        bpy.utils.unregister_class(PrevDipoleConnection)
    except:
        pass




