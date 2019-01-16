import bpy
import numpy as np
import os.path as op


def run(mmvt):
    mmvt.meg.plot_activity_contours('sample_audvis-meg_contoures_10', colormap='RdOrYl')
    # mu = mmvt.utils
    # mmvt.show_activity()
    # all_contours = mu.load(op.join(mu.get_user_fol(), 'meg', 'clusters', 'sample_audvis-meg_contoures_10.pkl'))
    # thresholds = sorted([float(k) for k in all_contours.keys()])
    # mmvt.colorbar.set_colormap('RdOrYl')
    # mmvt.colorbar.set_colorbar_max_min(np.max(thresholds), 0)
    # for hemi in mu.HEMIS:
    #     hemi_contours = np.zeros(len(bpy.data.objects[hemi].data.vertices))
    #     for threshold in all_contours.keys():
    #         hemi_contours[all_contours[threshold][hemi]] = threshold
    #     mesh = mu.get_hemi_obj(hemi).data
    #     mesh.vertex_colors.active_index = mesh.vertex_colors.keys().index('contours')
    #     mesh.vertex_colors['contours'].active_render = True
    #     mmvt.color_hemi_data(hemi, hemi_contours, 0.1, 256 / np.max(thresholds), override_current_mat=True,
    #                     coloring_layer='contours', check_valid_verts=False)
