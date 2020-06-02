import bpy
import os.path as op
from scripts_panel import ScriptsPanel


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    pass


def draw(self, context):
    layout = self.layout
    layout.operator(SelectElectrode.bl_idname, text="Select an Electorde", icon='FORCE_HARMONIC')
    if len(bpy.context.selected_objects) == 1 and \
            bpy.context.scene.move_elextrode_name == bpy.context.selected_objects[0].name:
        row = layout.row(align=True)
        row.label(text=bpy.context.selected_objects[0].name)
        row.prop(context.scene, 'move_elextrode_x')
        row.prop(context.scene, 'move_elextrode_y')
        row.prop(context.scene, 'move_elextrode_z')
        layout.operator(UpdateLead.bl_idname, text="Update the current lead", icon='SNAP_SURFACE')


def move_elec_update(self, context):
    mmvt = _mmvt()
    elc = bpy.context.selected_objects[0]
    coo = (bpy.context.scene.move_elextrode_x, bpy.context.scene.move_elextrode_y, bpy.context.scene.move_elextrode_z)
    mmvt.where_am_i.set_ct_coo(coo)
    tkreg_ras = mmvt.where_am_i.get_tkreg_ras()
    for k in range(3):
        elc.location[k] = tkreg_ras[k] * 0.1
    mmvt.where_am_i.create_slices(pos=tkreg_ras)


def update_lead(elc_name):
    elecs, mu = _mmvt().electrodes, _mmvt().utils
    group, _ = mu.elec_group_number(elc_name)
    lead_obj_name = '{}_lead'.format(group)
    ret = mu.delete_object(lead_obj_name)
    if not ret:
        return False
    electrodes = [o.name for o in bpy.data.objects['Deep_electrodes'].children if o.name.startswith(group)]
    electrodes = sorted([mu.elec_group_number(elec)[::-1] for elec in electrodes])
    electrodes = ['{}{}'.format(group, num) for num, group in electrodes]
    elecs.create_lead(elecs.get_elc_pos(electrodes[0]), elecs.get_elc_pos(electrodes[-1]), lead_obj_name)


def select_electrode(elc_name):
    mmvt = _mmvt()
    bpy.context.scene.move_elextrode_name = elc_name
    ct_vox = mmvt.where_am_i.get_ct_voxel()
    bpy.context.scene.move_elextrode_x = ct_vox[0]# bpy.data.objects[elc_name].location[0]
    bpy.context.scene.move_elextrode_y = ct_vox[1]# bpy.data.objects[elc_name].location[1]
    bpy.context.scene.move_elextrode_z = ct_vox[2]# bpy.data.objects[elc_name].location[2]


class SelectElectrode(bpy.types.Operator):
    bl_idname = "mmvt.select_electrode"
    bl_label = "mmvt select_electrode"
    bl_description = 'Select Electrode'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        select_electrode(bpy.context.active_object.name)
        return {"FINISHED"}


class UpdateLead(bpy.types.Operator):
    bl_idname = "mmvt.update_lead"
    bl_label = "mmvt update lead"
    bl_description = 'Update Lead'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        update_lead(bpy.context.active_object.name)
        return {"FINISHED"}


bpy.types.Scene.move_elextrode_x = bpy.props.IntProperty(default=0, step=1, name='x', update=move_elec_update)
bpy.types.Scene.move_elextrode_y = bpy.props.IntProperty(default=0, step=1, name='y', update=move_elec_update)
bpy.types.Scene.move_elextrode_z = bpy.props.IntProperty(default=0, step=1, name='z', update=move_elec_update)
bpy.types.Scene.move_elextrode_name = bpy.props.StringProperty()


def init(mmvt):
    mmvt.electrodes.color_the_relevant_lables(False)
    register()


def register():
    try:
        bpy.utils.register_class(SelectElectrode)
        bpy.utils.register_class(UpdateLead)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(SelectElectrode)
        bpy.utils.unregister_class(UpdateLead)
    except:
        pass

