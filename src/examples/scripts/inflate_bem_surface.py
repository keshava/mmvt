import bpy
import os.path as op
from scripts_panel import ScriptsPanel
import nibabel as nib


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    mu = mmvt.mmvt_utils
    surface_vertices, _ = mu.read_ply_file(op.join(
        mmvt.utils.get_user_fol(), 'surf', '{}.ply'.format(bpy.context.scene.bem_surfaces)))
    surface = bpy.data.objects[bpy.context.scene.bem_surfaces]
    for ind, vert in enumerate(surface.data.vertices):
        vert.co = surface_vertices[ind] + vert.normal * bpy.context.scene.inflate_bem_surface_factor


def surface_vis_update(self, context):
    mu.show_hide_obj(obj, val)


def export_surface():
    mu = _mmvt().mmvt_utils
    bem_fol = op.join(mu.get_subjects_dir(), mu.get_user_fol(), 'bem')
    backup_fol = mu.make_dir(op.join(bem_fol, 'backup'))
    for surf_name in bpy.types.Scene.bem_surfaces:
        surf_obj = bpy.data.objects.get(surf_name)
        if surf_obj is None:
            print('{} wasn\'t imported into MMVT!')
            continue
        surf_fname = op.join(bem_fol, '{}.fif'.format(surf_name))
        mu.copy_file(surf_fname, backup_fol)
        surf_vertices = [v.co for v in surf_obj.data.vertices]
        surf_faces = [f for f in surf_obj.faces]
        print('sdf')
        # nib.freesurfer.write_geometry(surf_fname, surf_vertices, surf_faces)


class ExportSurface(bpy.types.Operator):
    bl_idname = "mmvt.bem_export_surface"
    bl_label = "mmvt bem_export_surface"
    bl_description = 'Save image'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        export_surface()
        return {"FINISHED"}


bpy.types.Scene.inflate_bem_surface_factor = bpy.props.FloatProperty(default=0)
bpy.types.Scene.inflate_bem_brain_surface = bpy.props.BoolProperty(default=True, update=surface_vis_update)
bpy.types.Scene.inflate_bem_inner_skull_surface = bpy.props.BoolProperty(default=True, update=surface_vis_update)
bpy.types.Scene.inflate_bem_outer_skin_surface = bpy.props.BoolProperty(default=True, update=surface_vis_update)
bpy.types.Scene.inflate_bem_outer_skull_surface = bpy.props.BoolProperty(default=True, update=surface_vis_update)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'inflate_bem_brain_surface', 'Brain')
    layout.prop(context.scene, 'inflate_bem_inner_skull_surface', 'Inner Skull')
    layout.prop(context.scene, 'inflate_bem_outer_skull_surface', 'Outer Skull')
    layout.prop(context.scene, 'inflate_bem_outer_skin_surface', 'Outer Skin')
    layout.prop(context.scene, 'bem_surfaces', '')
    layout.prop(context.scene, 'inflate_bem_surface_factor', text='inf factor')
    layout.operator(ExportSurface.bl_idname, text="Export BEM", icon='FORCE_TURBULENCE')


def init(mmvt):
    register()
    surfaces = ['brain_surface', 'inner_skull_surface', 'outer_skin_surface', 'outer_skull_surface']
    bpy.types.Scene.bem_surfaces = bpy.props.EnumProperty(
        items=[(s, s.replace('_', ' '), '', c) for c, s in enumerate(surfaces)],
        description='BEM surfaces')
    bpy.context.scene.bem_surfaces = surfaces[0]


def register():
    try:
        bpy.utils.register_class(ExportSurface)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(ExportSurface)
    except:
        pass
