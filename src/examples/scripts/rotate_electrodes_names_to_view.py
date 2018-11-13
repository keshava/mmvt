import bpy

def run(mmvt):
    mu = mmvt.utils
    tmp = mu.get_3d_spaces(only_neuro=True)
    view_region = tmp.__next__().region_3d
    q=view_region.view_rotation.to_euler()
    bpy.context.scene.view_rotations_x = view_region.view_rotation.to_euler()[0]
    bpy.context.scene.view_rotations_y = view_region.view_rotation.to_euler()[1]
    bpy.context.scene.view_rotations_z = view_region.view_rotation.to_euler()[2]
    for obj in bpy.data.objects['texts'].children:
        print(obj.name)
        obj.rotation_euler[0]+=0.00000001
    print('done!')
