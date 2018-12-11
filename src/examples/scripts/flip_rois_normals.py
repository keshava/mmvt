import bpy
import mathutils
import numpy as np


def run(mmvt):
    mu = mmvt.utils
    for hemi in mu.HEMIS:
        parent = bpy.data.objects['Cortex-{}'.format(hemi)]
        hemi_obj = bpy.data.objects[hemi]
        size = len(hemi_obj.data.vertices)
        kd = mathutils.kdtree.KDTree(size)
        for vert_ind, vert in enumerate(hemi_obj.data.vertices):
            kd.insert(vert.co, vert_ind)
        kd.balance()
        for roi in parent.children:
            for vert in roi.data.vertices:
                co, index, dist = kd.find(vert.co)
                if dist == 0:
                    break
                    # print('dist!', roi.name, dist)
            else:
                print('asdfdf')
            hemi_normal = hemi_obj.data.vertices[index].normal
            ang = angle_between(hemi_normal, vert.normal)
            if ang > 3:
                print('flip!', roi.name, ang)
                roi.select = True
                # mu.fix_normals(roi)
                mu.recalc_normals(roi, True)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ang1 = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    ang2 = 2 * np.pi - ang1
    return min(ang1, ang2)

