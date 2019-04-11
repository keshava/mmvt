import bpy
import numpy as np
import time


def run(mmvt):
    mu = mmvt.utils
    parent = bpy.data.objects.get('Deep_electrodes', None)
    if parent is None:
        return

    bpy.ops.curve.primitive_bezier_curve_add()
    orig_curve = bpy.context.active_object
    orig_curve.name = 'org_curve'
    mu.create_and_set_material(orig_curve)

    conn_parent_name = mmvt.connections.get_connections_parent_name()
    mu.delete_hierarchy(conn_parent_name)
    mu.create_empty_if_doesnt_exists(conn_parent_name, mmvt.BRAIN_EMPTY_LAYER, None, 'Functional maps')
    conn_parent_obj = bpy.data.objects[conn_parent_name]
    layers_rods = [False] * 20
    layers_rods[mmvt.CONNECTIONS_LAYER] = True
    N = len(parent.children)
    con = np.full((N, N), False, dtype=bool)
    now = time.time()
    runs = int(N * N / 2)
    run_ind = 0
    for ind1, elc_obj1 in enumerate(parent.children):
        for ind2, elc_obj2 in enumerate(parent.children):
            if ind1 == ind2:
                continue
            if con[ind1, ind2] or con[ind2, ind1]:
                continue
            group1, _, _  = mu.elec_group_number(elc_obj1.name, True)
            group2, _, _ = mu.elec_group_number(elc_obj2.name, True)
            if group1 == group2:
                continue
            mu.time_to_go(now, run_ind, runs, 100)
            con[ind1, ind2] = True
            # mu.create_bezier_curve(elc_obj1, elc_obj2, layers_rods)
            cur_obj = copy_curve(
                orig_curve, elc_obj1, elc_obj2, conn_parent_name, layers_rods, bevel_depth=0.1, resolution_u=1)
            cur_obj.name = '{}-{}'.format(elc_obj1.name, elc_obj2.name)
            cur_obj.parent = conn_parent_obj
            run_ind += 1
            # mu.create_and_set_material(cur_obj)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[orig_curve.name].select = True
    bpy.ops.object.delete()


def copy_curve(orig_curve, obj1, obj2, parent_name, layers_array, bevel_depth=0.1, resolution_u=1):
    m = orig_curve.data.copy()
    obj = bpy.data.objects.new('curv', m)
    obj.location = (0, 0, 0)
    curve = obj.data
    curve.dimensions = '3D'
    curve.fill_mode = 'FULL'
    curve.splines[0].bezier_points[0].co = obj1.location
    curve.splines[0].bezier_points[1].co = obj2.location
    curve.bevel_depth = bevel_depth
    curve.resolution_u = resolution_u
    bpy.ops.object.move_to_layer(layers=layers_array)
    obj.parent = bpy.data.objects[parent_name]
    bpy.context.scene.objects.link(obj)
    return obj
