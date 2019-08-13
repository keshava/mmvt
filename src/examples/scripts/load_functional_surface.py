import bpy


def run(mmvt):
    ply_fname = mmvt.utils.get_real_fname('load_ply_object_fname')
    mmvt.data.load_ply(ply_fname, bpy.context.scene.load_ply_object_name, new_material_name='{}_mat'.format(
        bpy.context.scene.load_ply_object_name))


bpy.types.Scene.load_ply_object_fname = bpy.props.StringProperty(subtype='FILE_PATH')
bpy.types.Scene.load_ply_object_name = bpy.props.StringProperty()


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'load_ply_object_fname', text='ply fname')
    layout.prop(context.scene, 'load_ply_object_name', text='Object name')
