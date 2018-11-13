import bpy

def create_driver(obj,driver_type="rotation_euler",driver_value=0,target_property='view_rotations_x'):
    new_driver = obj.driver_add(driver_type,driver_value)
    new_driver.driver.type = 'AVERAGE'
    var = new_driver.driver.variables.new()
    var.type='SINGLE_PROP'
    target = var.targets[0]
    target.id_type = 'SCENE'
    target.id = bpy.data.scenes['Scene']
    target.data_path = target_property

def create_text(obj,parent_obj):
    bpy.ops.object.text_add(location=obj.location, rotation=(45,0,0))
    #print("test")
    text_obj = bpy.context.object
    text_obj.name = obj.name+'_text' #Objectname
    text_obj.rotation_euler.x = 0 # Rotate text by 90 degrees along X axis
    text_obj.data.body = obj.name
    text_obj.scale[0] = 0.25
    text_obj.scale[1] = 0.25
    text_obj.scale[2] = 0.25
    text_obj.active_material = bpy.data.materials['text_mat']
    text_obj.show_x_ray = True
    text_obj.parent = parent_obj
    bpy.context.active_object.active_material.diffuse_color = (0,0,0)

def create_material():
    curMat = None
    curMat = bpy.data.materials['OrigPatchMatTwoCols'].copy()
    curMat.name = 'text_mat'
    curMat.node_tree.nodes['MyColor'].inputs[0].default_value = (0,0,0,1)
    curMat.node_tree.nodes['MyColor1'].inputs[0].default_value = (0,0,0,1)
    curMat.node_tree.nodes['MyTransparency'].inputs['Fac'].default_value = 1
    curMat.use_nodes = False


def run(mmvt):
    mu = mmvt.utils
    bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    texts_obj = bpy.context.object
    texts_obj.name = 'texts' #Objectname
    create_material()
    for electrode_obj in bpy.data.objects['Deep_electrodes'].children:
        create_text(electrode_obj,texts_obj)
        text_obj = bpy.data.objects[electrode_obj.name+'_text']
        for ii,axis_str in enumerate(['x','y','z']):
            create_driver(text_obj,driver_type="rotation_euler",driver_value=ii,target_property='view_rotations_'+axis_str)
