import bpy
import os.path as op
from scripts_panel import ScriptsPanel


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    mu = mmvt.mmvt_utils
    surface_vertices, _ = mu.read_ply_file(op.join(
        mmvt.utils.get_user_fol(), 'surf', '{}.ply'.format(bpy.context.scene.bem_surfaces)))
    surface = bpy.data.objects[bpy.context.scene.bem_surfaces]
    for ind, vert in enumerate(surface.data.vertices):
        vert.co = surface_vertices[ind] + vert.normal * bpy.context.scene.inflate_bem_surface_factor


def export_surface():
    pass


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


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'bem_surfaces', '')
    layout.prop(context.scene, 'inflate_bem_surface_factor', text='inf factor')


def init(mmvt):
    surfaces = ['brain_surface', 'inner_skull_surface', 'outer_skin_surface', 'outer_skull_surface']
    bpy.types.Scene.bem_surfaces = bpy.props.EnumProperty(
        items=[(s, s.replace('_', ' '), '', c) for c, s in enumerate(surfaces)],
        description='BEM surfaces')
    bpy.context.scene.bem_surfaces = surfaces[0]
