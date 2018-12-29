import bpy


def run(mmvt):
    selected_objects = bpy.context.selected_objects
    if len(selected_objects) == 0:
        return
    cur_obj = bpy.context.selected_objects[0]
    mmvt.coloring.object_coloring(cur_obj, bpy.context.scene.selected_object_color)


def draw(self, context):
    layout = self.layout
    selected_objects = bpy.context.selected_objects
    if len(selected_objects) == 0:
        return
    # Color picker:
    row = layout.row(align=True)
    layout.label(text=selected_objects[0].name)
    row.label(text='Selected color')
    row.prop(context.scene, 'selected_object_color', text='')



bpy.types.Scene.selected_object_color = bpy.props.FloatVectorProperty(
    name="selected_object_color", subtype='COLOR', default=(0, 0.5, 0), min=0.0, max=1.0)
