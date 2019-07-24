import numpy as np
import os.path as op
from collections import Counter
from itertools import product, combinations
# from sklearn.preprocessing import StandardScaler
# from sklearn import svm
# from sklearn.externals import joblib
from scipy.spatial.distance import cdist
from queue import Empty
import time
import nibabel as nib
import glob


def find_voxels_above_threshold(ct_data, threshold):
    return np.array(np.where(ct_data > threshold)).T


def ct_voxels_to_t1_tkreg(voxels, ct_header, brain):
    if len(voxels) == 0:
        return []
    brain_header = brain.get_header()
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    ct_ras = apply_trans(ct_vox2ras, voxels)
    t1_vox = np.rint(apply_trans(ras2t1_vox, ct_ras)).astype(int)
    t1_tkreg = apply_trans(vox2t1_ras_tkr, t1_vox)
    return t1_tkreg


def mask_voxels_outside_brain(voxels, ct_header, brain, subject_mmvt_fol, subject_fs_fol, sigma):
    if len(voxels) == 0:
        return [], []
    # brain_header = brain.get_header()
    # # brain_mask = brain.get_data()
    # ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    # ct_ras = apply_trans(ct_vox2ras, voxels)
    # t1_vox = np.rint(apply_trans(ras2t1_vox, ct_ras)).astype(int)
    # t1_tkreg = apply_trans(vox2t1_ras_tkr, t1_vox)
    brain_header = brain.get_header()
    t1_tkreg = ct_voxels_to_t1_tkreg(voxels, ct_header, brain)
    t1_voxels_inside_dural, voxels_indices = get_t1_voxels_inside_dural(
        t1_tkreg, brain_header, subject_mmvt_fol, subject_fs_fol, False, sigma)

    t1_ras_inside_dural = apply_trans(brain_header.get_vox2ras(), t1_voxels_inside_dural)
    ct_voxels_inside_dural = np.rint(apply_trans(np.linalg.inv(ct_header.get_vox2ras()), t1_ras_inside_dural)).astype(int)

    # for vox in ct_voxels_inside_dural:
    #     if vox[0] >= ct_data_shape[0] or vox[1] >= ct_data_shape[1] or vox[2] >= ct_data_shape[2]:
    #         print('vox outside the data! {}'.format(vox))

    # if vertices is None:
    #     print('Loading pial vertices for finding electrodes hemis')
    #     verts = read_pial_verts(user_fol)
    #     vertices = np.concatenate((verts['rh'], verts['lh']))
    # if aseg is not None:
    #     # t1_voxels_outside_pial = get_t1_voxels_outside_pial(
    #     #     brain_header, brain_mask, aseg, t1_tkreg, t1_vox, vertices, sigma)
    #     t1_voxels_outside_pial = get_t1_voxels_outside_dural(t1_tkreg, brain_header, subject_fol, sigma=0)
    #     t1_voxels_outside_pial = set([tuple(v) for v in t1_voxels_outside_pial])
    #     tup = [(ind, v) for ind, (v, t1_v) in enumerate(zip(voxels, t1_vox)) if
    #            aseg[tuple(t1_v)] not in [7, 8] and tuple(t1_v) not in t1_voxels_outside_pial]
    #     # tup = [(ind, v) for ind, (v, t1_v) in enumerate(zip(voxels, t1_vox)) if brain_mask[tuple(t1_v)] > 0 and
    #     #        aseg[tuple(t1_v)] not in [7, 8] and tuple(t1_v) not in t1_voxels_outside_pial]
    # else:
    #     tup = [(ind, v) for ind, (v, t1_v) in enumerate(zip(voxels, t1_vox)) if brain_mask[tuple(t1_v)] > 0]
    # voxels = np.array([t[1] for t in tup])
    # voxels_indices = np.array([t[0] for t in tup])
    return ct_voxels_inside_dural, voxels_indices


def get_t1_voxels_outside_pial(brain_header, brain_mask, aseg, t1_tkreg, t1_vox, vertices, sigma=2):
    if t1_vox.dtype != np.dtype(int):
        t1_vox = np.rint(t1_vox).astype(int)
    voxel_outside_pial = np.array([t1_v for t1_v, t1_t in zip(t1_vox, t1_tkreg) if brain_mask[tuple(t1_v)] == 0])
    return voxel_outside_pial


def get_t1_voxels_inside_dural(t1_tkreg, brain_header, subject_mmvt_fol, subject_fs_fol, use_brain_surf=True,
                               sigma=1, sigma_in=0):
    dural_mask_hemi, close_verts_dists = {}, {}
    hemis = ['lh', 'rh']
    # if use_brain_surf:
    #     brain_verts = get_brain_surface(subject_mmvt_fol)
    #     if brain_verts is None:
    #         print("Can't find the brain surface! No bem solution was found! You should use mne_watershed_bem.")
    #     use_brain_surf = brain_verts is not None
    verts, faces, normals = get_dural_surface(subject_mmvt_fol, subject_fs_fol, do_calc_normals=True)
    if verts is None or faces is None:
        raise Exception("No dural surface!")
    import bpy
    for hemi_ind, hemi in enumerate(hemis):
        # todo: check the normals vs Blender normals
        # normals = calc_normals(verts[hemi], faces[hemi])
        dural_surf = bpy.data.objects['dural-{}'.format(hemi)]
        # verts[hemi] = np.array([v.co for v in dural_surf.data.vertices])
        # normals[hemi] = np.array([v.normal for v in dural_surf.data.vertices])
        # dists = cdist(t1_tkreg, verts[hemi])
        # if use_brain_surf:
        #     close_verts_dists[hemi] = np.min(dists, axis=1)
        # close_verts_indices = np.argmin(dists, axis=1)
        # dural_mask_hemi[hemi] = [point_in_mesh(u, verts[hemi][vert_ind], normals[hemi][vert_ind], sigma, sigma_in)
        #                          for u, vert_ind in zip(t1_tkreg, close_verts_indices)]
        dural_mask_hemi[hemi] = [_is_inside(u, dural_surf) for u in t1_tkreg]

    mask = np.array([dural_mask_hemi[hemi] for hemi in hemis]).T
    if use_brain_surf:
        mask = mask_dural_with_brain(t1_tkreg, mask, brain_verts, close_verts_dists)
    indices = np.unique(np.where(mask)[0])
    voxel_inside_pial = np.rint(
        apply_trans(np.linalg.inv(brain_header.get_vox2ras_tkr()), t1_tkreg[indices])).astype(int)
    return voxel_inside_pial, indices


def _is_inside(p, obj):
    from mathutils import Vector

    p = Vector(p)
    res, point, normal, face = obj.closest_point_on_mesh(p)
    p2 = point-p
    v = p2.dot(normal)
    # print(v)
    return not(v < 0.0)

def mask_dural_with_brain(t1_tkreg, dural_mask, brain_verts, close_verts_dists):
    brain_dists = cdist(t1_tkreg, brain_verts)
    close_brain_dists = np.min(brain_dists, axis=1)
    close_verts_dists = np.array([close_verts_dists[hemi] for hemi in ['lh', 'rh']]).T
    mask = [any(in_dural) or max(v_dists) < b_dist for v_dists, b_dist, in_dural in
            zip(close_verts_dists, close_brain_dists, dural_mask)]
    return mask


# def get_t1_voxels_outside_pial(brain_header, brain_mask, aseg, t1_tkreg, t1_vox, vertices, sigma=2):
#     if t1_vox.dtype != np.dtype(int):
#         t1_vox = np.rint(t1_vox).astype(int)
#     electrodes = np.array([t1_t for t1_v, t1_t in zip(t1_vox, t1_tkreg)
#                            if brain_mask[tuple(t1_v)] > 0 and aseg[tuple(t1_v)] == 0])
#     if len(electrodes) == 0:
#         return [], []
#     dists = cdist(electrodes, vertices)
#     close_verts = np.argmin(dists, axis=1)
#     print('Masking voxels outside the brain with sigma={}'.format(sigma))
#     outside_pial = [u for u, v in zip(electrodes, close_verts) if np.linalg.norm(u) + sigma > np.linalg.norm(vertices[v])]
#     voxel_outside_pial = apply_trans(np.linalg.inv(brain_header.get_vox2ras_tkr()), outside_pial)
#     return voxel_outside_pial, close_verts


def get_trans(ct_header, brain_header):
    ct_vox2ras = ct_header.get_vox2ras()
    ras2t1_vox = np.linalg.inv(brain_header.get_vox2ras())
    vox2t1_ras_tkr = brain_header.get_vox2ras_tkr()
    return ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr


def find_all_local_maxima(ct_data, voxels, threshold=0, find_nei_maxima=False, max_iters=100):
    maxs = set()
    for run, voxel in enumerate(voxels):
        if find_nei_maxima:
            max_voxel = find_local_nei_maxima_in_ct(ct_data, voxel, threshold, max_iters)
        else:
            max_voxel = find_local_maxima_in_ct(ct_data, voxel, max_iters)
        if max_voxel is not None:
            maxs.add(tuple(max_voxel))
    maxs = np.array([np.array(vox) for vox in maxs])
    return maxs


def remove_neighbors_voxels(ct_data, voxels):
    if len(voxels) == 0:
        return []
    dists = cdist(voxels, voxels, 'cityblock')
    inds = np.where(dists == 1)
    if len(inds[0]) > 0:
        pairs = list(set([tuple(sorted([inds[0][k], inds[1][k]])) for k in range(len(inds[0]))]))
        to_remove = [pair[0] if ct_data[tuple(voxels[pair[0]])] < ct_data[tuple(voxels[pair[1]])]
                     else pair[1] for pair in pairs]
        voxels = np.delete(voxels, to_remove, axis=0)
    return voxels


def find_local_maxima_in_ct(ct_data, voxel, max_iters=100):
    peak_found, iter_num = False, 0
    voxel_max = voxel.copy()
    while not peak_found and iter_num < max_iters:
        max_ct_data = ct_data[tuple(voxel_max)]
        max_diffs = (0, 0, 0)
        neighbors = get_voxel_neighbors_ct_values(ct_data, voxel_max, True)
        for ct_value, delta in neighbors:
            if ct_value > max_ct_data:
                max_ct_data = ct_value
                max_diffs = delta
        peak_found = max_diffs == (0, 0, 0)
        voxel_max += max_diffs
        iter_num += 1
    if not peak_found:
        # print('Peak was not found!')
        voxel_max = None
    return voxel_max


def find_local_nei_maxima_in_ct(ct_data, voxel, threshold=0, max_iters=100):
    peak_found, iter_num = False, 0
    voxel_max = voxel.copy()
    while not peak_found and iter_num < max_iters:
        max_nei_ct_data = sum(get_voxel_neighbors_ct_values(ct_data, voxel_max, False))
        max_diffs = (0, 0, 0)
        neighbors = get_voxel_neighbors_ct_values(ct_data, voxel_max, True)
        for ct_val, delta in neighbors:
            neighbors_neighbors_ct_val = sum(get_voxel_neighbors_ct_values(ct_data, voxel+delta, False))
            if neighbors_neighbors_ct_val > max_nei_ct_data and ct_val > threshold:
                max_nei_ct_data = neighbors_neighbors_ct_val
                max_diffs = delta
        peak_found = max_diffs == (0, 0, 0)
        voxel_max += max_diffs
        iter_num += 1
    if not peak_found:
        # print('Peak was not found!')
        voxel_max = None
    return voxel_max


def get_voxel_neighbors_ct_values(ct_data, voxel, include_new_voxel=False, r=1):
    x, y, z = np.rint(voxel).astype(int)
    if include_new_voxel:
        return [(ct_data[x + dx, y + dy, z + dz], (dx, dy, dz))
                for dx, dy, dz in product([-r, 0, r], [-r, 0, r], [-r, 0, r]) if
                in_shape((x+dx, y+dy, z+dz), ct_data.shape)]
                # 0 <= x + dx < ct_data.shape[0] and 0 <= y + dy < ct_data.shape[1] and 0 <= z + dz < ct_data.shape[2]]
    else:
        return ct_data[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1].ravel().tolist()
        # return [ct_data[x + dx, y + dy, z + dz] for dx, dy, dz in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]) if
        #         0 <= x + dx < ct_data.shape[0] and 0 <= y + dy < ct_data.shape[1] and 0 <= z + dz < ct_data.shape[2]]


# def find_electrodes_hemis_svm(user_fol, electrodes, groups=None, overwrite=False):
#     output_fname = op.join(user_fol, 'ct', 'finding_electrodes_in_ct', 'hemis_model.pkl')
#     if not op.isfile(output_fname) or overwrite:
#         print('Loading pial vertices for finding electrodes hemis')
#         verts = read_surf_verts(user_fol)
#         vertices = np.concatenate((verts['rh'], verts['lh']))
#         hemis = np.array([0] * len(verts['rh']) + [1] * len(verts['lh']))
#         scaler = StandardScaler()
#         vertices = scaler.fit_transform(vertices)
#         electrodes = scaler.fit_transform(electrodes)
#         clf = svm.SVC(kernel='linear')
#         print('Fitting the vertices...')
#         clf.fit(vertices, hemis)
#         joblib.dump(clf, output_fname, compress=9)
#     else:
#         clf = joblib.load(output_fname)
#     elctrodes_hemis = clf.predict(electrodes)
#     hemis = ['rh' if elc_hemi == 0 else 'lh' for elc_hemi in elctrodes_hemis]
#     if groups is not None:
#         groups_hemis = [Counter([hemis[elc] for elc in group]).most_common()[0][0] for group in groups]
#     else:
#         groups_hemis = []
#     return hemis, groups_hemis
#

def find_electrodes_hemis(subject_fol, electrodes_t1_tkreg, groups=None, sigma=0, dural_verts=None, dural_normals=None):
    in_dural = {}
    if dural_verts is None or dural_normals is None:
        dural_verts, _, dural_normals = get_dural_surface(subject_fol, do_calc_normals=True)
    for hemi in ['rh', 'lh']:
        dists = cdist(electrodes_t1_tkreg, dural_verts[hemi])
        close_verts_indices = np.argmin(dists, axis=1)
        in_dural[hemi] = [point_in_mesh(u, dural_verts[hemi][vert_ind], dural_normals[hemi][vert_ind], sigma)
                          for u, vert_ind in zip(electrodes_t1_tkreg, close_verts_indices)]
    hemis = ['rh' if in_dural['rh'][ind] else 'lh' if in_dural['lh'][ind] else 'un' for ind in
             range(len(electrodes_t1_tkreg))]
    if groups is not None:
        groups_hemis = [Counter([hemis[elc] for elc in group]).most_common()[0][0] for group in groups]
        return hemis, groups_hemis
    else:
        return hemis


def find_group_between_pair(elc_ind1, elc_ind2, electrodes, error_radius=3, min_distance=2):
    points_inside, cylinder, dists_to_cylinder = points_in_cylinder(
        electrodes[elc_ind1], electrodes[elc_ind2], electrodes, error_radius)
    sort_indices = np.argsort([np.linalg.norm(electrodes[p] - electrodes[elc_ind1]) for p in points_inside])
    points_inside = [points_inside[ind] for ind in sort_indices]
    dists_to_cylinder = dists_to_cylinder[sort_indices]
    # points_inside_before_removing_too_close_points = points_inside.copy()
    # print('remove_too_close_points', points_inside, cylinder[0], cylinder[-1], min_distance)
    points_inside, too_close_points, removed_indices = remove_too_close_points(
        electrodes, points_inside, cylinder, min_distance)
    group = points_inside.tolist() if not isinstance(points_inside, list) else points_inside.copy()
    dists = calc_group_dists(electrodes[points_inside])
    gof = np.mean(dists_to_cylinder)
    dists_to_cylinder = np.delete(dists_to_cylinder, removed_indices)
    dists_to_cylinder = {p:d for p, d in zip(points_inside, dists_to_cylinder)}
    return group, too_close_points, dists, dists_to_cylinder, gof, elc_ind1


def find_electrode_group(elc_ind, electrodes, elctrodes_hemis, groups=[], error_radius=3, min_elcs_for_lead=4,
                         max_dist_between_electrodes=15, min_distance=2, find_better_groups=False):
    max_electrodes_inside = 0
    # best_points_insides, best_group_too_close_points, best_group_dists, best_dists_to_cylinder = [], [], [], []
    best_groups = {}
    # best_cylinder = None
    elcs_already_in_groups = set(flat_list_of_lists(groups))
    electrodes_list = list(set(range(len(electrodes))) - elcs_already_in_groups)
    combs = combinations(electrodes_list, 2)
    print('Finding group for elc {}'.format(elc_ind))
    # for i in electrodes_list:
    #     for j in electrodes_list[i+1:]:
    for i, j in combs:
        points_inside, cylinder, dists_to_cylinder = find_elc_cylinder(
            electrodes, elc_ind, i, j, elcs_already_in_groups, elctrodes_hemis, min_elcs_for_lead, error_radius, min_distance)
        if points_inside is None:
            continue
        sort_indices = np.argsort([np.linalg.norm(electrodes[p] - electrodes[i]) for p in points_inside])
        points_inside = [points_inside[ind] for ind in sort_indices]
        elcs_inside = electrodes[points_inside]
        dists_to_cylinder = dists_to_cylinder[sort_indices]
        # elcs_inside = sorted(elcs_inside, key=lambda x: np.linalg.norm(x - electrodes[i]))
        dists = calc_group_dists(elcs_inside)
        if max(dists) > max_dist_between_electrodes:
            continue
        # points_inside_before_removing_too_close_points = points_inside.copy()
        points_inside, too_close_points, removed_indices = remove_too_close_points(
            electrodes, points_inside, cylinder, min_distance)
        dists_to_cylinder = np.delete(dists_to_cylinder, removed_indices)
        if len(points_inside) < min_elcs_for_lead:
            continue
        gof = np.mean(dists_to_cylinder)
        if len(points_inside) >= max_electrodes_inside:
            if len(points_inside) > max_electrodes_inside:
                best_groups = {}
            max_electrodes_inside = len(points_inside)
            group_dists = calc_group_dists(electrodes[points_inside])
            dists_to_cylinder_dict = {p: d for p, d in zip(points_inside, dists_to_cylinder)}
            # best_group = points_insides.tolist() # if not isinstance(best_points_insides, list) else best_points_insides.copy()
            best_groups[gof] = (points_inside, too_close_points, group_dists, dists_to_cylinder_dict, gof, elc_ind)

            # best_points_inside_before_removing_too_close_points = points_inside_before_removing_too_close_points.copy()
            # best_cylinder = cylinder
    # For debug only
    # print('remove_too_close_points', best_points_inside_before_removing_too_close_points, best_cylinder[0], best_cylinder[-1], min_distance)
    # best_points_insides, best_group_too_close_points = remove_too_close_points(
    #     electrodes, best_points_inside_before_removing_too_close_points, best_cylinder, min_distance)
    # best_points_insides, connected_points = remove_connected_points(
    #     electrodes, best_points_insides, best_cylinder, ct_data, threshold, ct_header, brain_header)
    # best_group_too_close_points = np.concatenate((best_group_too_close_points, connected_points))

    # best_group = best_points_insides.tolist() if not isinstance(best_points_insides, list) else best_points_insides.copy()
    # return best_group, best_group_too_close_points, best_group_dists, best_dists_to_cylinder
    gofs = sorted(list(best_groups.keys()))
    if len(gofs) == 0:
        return [], [], [], [], [], 0
    print('found {} groups with {} electrodes, gofs {}'.format(
        len(gofs), max_electrodes_inside, ','.join(['{:.4f}'.format(gof) for gof in gofs])))
    if find_better_groups:
        group, noise, group_dists, dists_to_cylinder, gof, _ = best_groups[gofs[0]]
        return find_better_groups_post_search(
            elc_ind, group, noise, group_dists, dists_to_cylinder, gof, electrodes,
            elctrodes_hemis, groups, error_radius, max_dist_between_electrodes, min_distance)
    else:
        return best_groups[gofs[0]]


def find_elc_cylinder(electrodes, elc_ind, i, j, elcs_already_in_groups, elctrodes_hemis, min_elcs_for_lead,
                      error_radius, min_distance):
    # hemis might be 'un' and we are ok with that
    if (elctrodes_hemis[i] == 'rh' and elctrodes_hemis[j] == 'lh' or
            elctrodes_hemis[i] == 'lh' and elctrodes_hemis[j] == 'rh'):
        # print('Electrodes {} and {} are not in the same hemi!'.format(i, j))
        return None, None, None
    if not point_in_cube(electrodes[i], electrodes[j], electrodes[elc_ind], error_radius):
        return None, None, None
    elcs_dist = np.linalg.norm(electrodes[i] - electrodes[j])
    if elcs_dist < min_distance * min_elcs_for_lead:
        # print('elcs {} and {} dist is {} < {}'.format(i, j, elcs_dist, min_distance * min_elcs_for_lead))
        return None, None, None
    points_inside, cylinder, dists_to_cylinder = points_in_cylinder(
        electrodes[i], electrodes[j], electrodes, error_radius)
    if elc_ind not in points_inside:
        return None, None, None
    if len(set(points_inside) & elcs_already_in_groups) > 0:
        return None, None, None
    if len(points_inside) < min_elcs_for_lead:
        return None, None, None
    same_hemi = all_items_equall([elctrodes_hemis[p] for p in points_inside])
    if not same_hemi:
        return None, None, None
    return points_inside, cylinder, dists_to_cylinder


def point_in_cube(pt1, pt2, k, e=0):
    def p_in_the_middle(x, y, z, e=0):
        return x + e >= z >= y - e if x > y else x - e <= z <= y + e
    return p_in_the_middle(pt1[0], pt2[0], k[0], e) and p_in_the_middle(pt1[1], pt2[1], k[1], e) and \
           p_in_the_middle(pt1[2], pt2[2], k[2], e)


def points_in_cylinder(pt1, pt2, points, radius_sq, N=100, metric='euclidean'):
    dist = np.linalg.norm(pt1 - pt2)
    elc_ori = (pt2 - pt1) / dist # norm(elc_ori)=1mm
    # elc_line = np.array([pt1 + elc_ori*t for t in np.linspace(0, dist, N)])
    elc_line = (pt1.reshape(3, 1) + elc_ori.reshape(3, 1) @ np.linspace(0, dist, N).reshape(1, N)).T
    dists = np.min(cdist(elc_line, points, metric), 0)
    points_inside_cylinder = np.where(dists <= radius_sq)[0]
    return points_inside_cylinder, elc_line, dists[points_inside_cylinder]



def find_better_groups_post_search(original_elc_ind, group, noise, group_dists, dists_to_cylinder, gof, electrodes,
        hemis, groups, error_radius, max_dist_between_electrodes, min_distance):
    best_groups, best_group_len = {}, len(group)
    best_groups[gof] = (group, noise, group_dists, dists_to_cylinder, gof)
    for elc_ind in group:
        if elc_ind == original_elc_ind:
            continue
        group, noise, dists, dists_to_cylinder, gof, _ = find_electrode_group(
            elc_ind, electrodes, hemis, groups, error_radius, len(group), max_dist_between_electrodes, min_distance,
            find_better_groups=False)
        if len(group) >= best_group_len:
            if len(group) > best_group_len:
                print('Found a better group with {} electrodes!'.format(len(group)))
                best_groups = {}
                best_group_len = len(group)
            best_groups[gof] = (group, noise, dists, dists_to_cylinder, gof, elc_ind)
    gofs = sorted(list(best_groups.keys()))
    if len(gofs) > 0:
        print('found {} groups with gofs {}'.format(len(gofs), gofs))
        return best_groups[gofs[0]]
    else:
        return [], [], [], [], [], 0


def remove_too_close_points(electrodes, points_inside_cylinder, cylinder, min_distance):
    # Remove too close points
    points_examined, points_to_remove = set(), set()
    elcs_inside = electrodes[points_inside_cylinder]
    elcs_on = find_closest_points_on_cylinder(electrodes, points_inside_cylinder, cylinder)
    dists = cdist(elcs_on, elcs_on)
    dists += np.eye(len(elcs_on)) * min_distance * 2
    inds = np.where(dists < min_distance)
    if len(inds[0]) > 0:
        pairs = list(set([tuple(sorted([inds[0][k], inds[1][k]])) for k in range(len(inds[0]))]))
        # print('remove_too_close_points: {}'.format(pairs))
        pairs_electrodes = [[elcs_inside[p[k]] for k in range(2)] for p in pairs]
        for pair_electrode, pair in zip(pairs_electrodes, pairs):
            pair_dist_to_cylinder = np.min(cdist(np.array(pair_electrode), cylinder), axis=1)
            ind = np.argmax(pair_dist_to_cylinder)
            if pair[1 - ind] not in points_to_remove:
                points_to_remove.add(pair[ind])
    points_to_remove = list(points_to_remove)
    rectangles = find_triangles_in_group(electrodes, points_inside_cylinder, points_to_remove)
    points_to_remove.extend([points_inside_cylinder.index(r) for r in rectangles])
    # print('points_to_remove:', points_to_remove)
    elecs_to_remove = np.array(points_inside_cylinder)[points_to_remove]

    # Check if one of the removed points should be un-removed (calc dists)
    # points_inside_cylinder_after_removing = np.delete(points_inside_cylinder, points_to_remove, axis=0)
    # dists = cdist(electrodes[elecs_to_remove], electrodes[points_inside_cylinder_after_removing])
    # points_to_unremove = np.array(points_to_remove)[np.min(dists, axis=1) > min_distance]
    # points_to_remove = [e for e in points_to_remove if e not in points_to_unremove]
    # elecs_to_remove = np.array(points_inside_cylinder)[points_to_remove]

    if len(points_to_remove) > 0:
        points_inside_cylinder = np.delete(points_inside_cylinder, points_to_remove, axis=0)
    return points_inside_cylinder, elecs_to_remove, points_to_remove


def find_triangles_in_group(electrodes, points_inside_cylinder, point_removed, ratio=0.7):
    points_inside_cylinder = [e for e in points_inside_cylinder if e not in
                              set([points_inside_cylinder[k] for k in point_removed])]
    elcs_inside = electrodes[points_inside_cylinder]
    points_to_remove = []
    dists = cdist(elcs_inside, elcs_inside)
    for ind in range(len(dists) - 2):
        if dists[ind, ind+2] < (dists[ind, ind+1] + dists[ind+1, ind+2]) * ratio:
            points_to_remove.append(points_inside_cylinder[ind+1])
    return points_to_remove


def find_closest_points_on_cylinder(electrodes, points_inside_cylinder, cylinder):
    return cylinder[np.argmin(cdist(electrodes[points_inside_cylinder], cylinder), axis=1)]


def calc_group_dists(electrodes_group):
    return [np.linalg.norm(pt2 - pt1) for pt1, pt2 in zip(electrodes_group[:-1], electrodes_group[1:])]


def point_in_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const


def find_points_path_on_dural_surface(elc1_ind, elc2_ind, hemis, electrodes, dural_verts, dural_verts_nei, names,
                                      sigma=5, ang_thresholds=(0.1, 0.5), debug=False):

    def calc_closest_dural_vertices_indices(vert1_ind, electrodes_dural_indices, points, hemi, all_names):
        dists = cdist([dural_verts[hemi][vert1_ind]], dural_verts[hemi][electrodes_dural_indices])
        mins_dists = np.sort(dists[0])[:5]
        to_big_min_dists = np.where(mins_dists > np.min(mins_dists) * 2)[0]
        nei_num = 5 - len(to_big_min_dists)
        closest_dural_vertices_indices = np.argsort(dists[0])[:nei_num]
        if debug:
            print('{} closest_dural_vertices_indices'.format(len(closest_dural_vertices_indices)))
        if len(points) > 1:
            angs = np.zeros((len(closest_dural_vertices_indices)))
            for k, ind in enumerate(closest_dural_vertices_indices):
                pt = dural_verts[hemi][electrodes_dural_indices[ind]]
                ang = angle_between(pt - points[-1], points[-1] - points[-2])
                ang = min(np.pi - ang, ang)
                angs[k] = ang
                if debug:
                    print(all_names[ind],  ang)
            angs_sort = np.sort(angs)
            if angs_sort[0] < ang_thresholds[1] and angs_sort[1] > ang_thresholds[1] or \
                    angs_sort[0] < ang_thresholds[0]:
                closest_dural_vertices_indices = [closest_dural_vertices_indices[np.argmin(angs)]]
                if debug:
                    print('next electrodes will be {}'.format(all_names[closest_dural_vertices_indices[0]]))
        return closest_dural_vertices_indices

    def main(elc1_ind, elc2_ind):
        pt1, pt2 = electrodes[elc1_ind], electrodes[elc2_ind]
        hemi1, hemi2 = hemis[elc1_ind], hemis[elc2_ind]
        if hemi1 != hemi2:
            print('The electrodes should be in the same hemi!')
            return []
        hemi = hemi1
        dists = cdist(electrodes, dural_verts[hemi])
        electrodes_dural_indices = np.argmin(dists, 1)
        electrodes_dural_indices = np.delete(electrodes_dural_indices, elc1_ind, 0)
        all_names = np.delete(names, elc1_ind)
        points, indices = [pt1], [elc1_ind]
        dists = cdist([pt1, pt2], dural_verts[hemi])
        vert1_ind, vert2_ind = np.argmin(dists, 1)
        closest_dural_vertices_indices = calc_closest_dural_vertices_indices(
            vert1_ind, electrodes_dural_indices, points, hemi, all_names)
        vert1_nei = dural_verts_nei[hemi][vert1_ind]
        last_dist, last_argmin, switch_dist = np.inf, 0, 0
        while elc2_ind not in indices:
            # Find the neighbor how is closest to the final point
            dists = cdist([dural_verts[hemi][vert2_ind]], dural_verts[hemi][vert1_nei])
            vert1_ind = vert1_nei[np.argmin(dists)]
            dists = cdist([dural_verts[hemi][vert1_ind]], dural_verts[hemi][electrodes_dural_indices[closest_dural_vertices_indices]])
            min_dist = np.min(dists)
            argmin = np.argmin(dists)
            if debug:
                print(min_dist, all_names[closest_dural_vertices_indices[argmin]])
            if min_dist > last_dist:
                switch_dist += 1
                if debug:
                    print('min_dist > last_dist!')
            if ((min_dist > last_dist or min_dist == 0) and last_dist < sigma) or switch_dist > 10:
                new_elc_ind = closest_dural_vertices_indices[last_argmin]
                vert1_ind = electrodes_dural_indices[new_elc_ind]
                points.append(dural_verts[hemi][vert1_ind])
                indices.append(names.index(all_names[new_elc_ind]))
                if debug:
                    print('Found an electrode! {}'.format(all_names[new_elc_ind]))
                all_names = np.delete(all_names, new_elc_ind)
                electrodes_dural_indices = np.delete(electrodes_dural_indices, new_elc_ind, 0)
                closest_dural_vertices_indices = calc_closest_dural_vertices_indices(
                    vert1_ind, electrodes_dural_indices, points, hemi, all_names)
                last_dist = np.inf
                switch_dist = 0
            else:
                last_dist = min_dist
                last_argmin = argmin
            vert1_nei = dural_verts_nei[hemi][vert1_ind]
        return np.array(points), np.array(indices)

    points1, indices1 = main(elc1_ind, elc2_ind)
    points2, indices2 = main(elc2_ind, elc1_ind)
    if len(points1) != len(points2):
        ind = 1 if len(points1) < len(points2) else 2
    else:
        ind = 1 if calc_points_length(points1) < calc_points_length(points2) else 2
    return (points1, indices1) if ind == 1 else (points2[::-1], indices2[::-1])

############# Utils ##############


def ct_voxels_to_t1_ras_tkr(ct_voxels, ct_header, brain_header):
    if isinstance(ct_voxels, list):
        ct_voxels = np.array(ct_voxels)
    if ct_voxels.dtype != np.dtype(int):
        ct_voxels = np.rint(ct_voxels).astype(int)
    ndim = ct_voxels.ndim
    if ndim == 1:
        ct_voxels = np.array([ct_voxels])
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    centroids_ras = apply_trans(ct_vox2ras, ct_voxels)
    centroids_t1_vox = np.rint(apply_trans(ras2t1_vox, centroids_ras)).astype(int)
    centroids_t1_ras_tkr = apply_trans(vox2t1_ras_tkr, centroids_t1_vox)
    if ndim == 1:
        centroids_t1_ras_tkr = centroids_t1_ras_tkr[0]
    return centroids_t1_ras_tkr


def t1_ras_tkr_to_ct_voxels(t1_tkras_coords, ct_header, brain_header):
    if isinstance(t1_tkras_coords, list):
        t1_tkras_coords = np.array(t1_tkras_coords)
    ndim = t1_tkras_coords.ndim
    if ndim == 1:
        t1_tkras_coords = np.array([t1_tkras_coords])
    t1_vox = np.rint(apply_trans(np.linalg.inv(brain_header.get_vox2ras_tkr()), t1_tkras_coords)).astype(int)
    t1_ras = apply_trans(brain_header.get_vox2ras(), t1_vox)
    ct_vox = np.rint(apply_trans(np.linalg.inv(ct_header.get_vox2ras()), t1_ras)).astype(int)
    if ndim == 1:
        ct_vox = ct_vox[0]
    return ct_vox


def read_surf_verts(subject_mmvt_fol, subject_fs_fol, surf='pial', return_faces=False):
    verts, faces = {'rh':None, 'lh':None}, {'rh':None, 'lh':None}
    for hemi in ['rh', 'lh']:
        fs_surf_fname = op.join(subject_fs_fol, 'surf', '{}.{}'.format(hemi, surf))
        if op.isfile(fs_surf_fname):
            verts[hemi], faces[hemi] = nib.freesurfer.read_geometry(fs_surf_fname)
            continue
        npz_surf_fname = op.join(subject_mmvt_fol, 'surf', '{}.{}.npz'.format(hemi, surf))
        if op.isfile(npz_surf_fname):
            d = np.load(npz_surf_fname)
            verts[hemi], faces[hemi] = d['verts'], d['faces']
            continue
        ply_surf_fname = op.join(subject_mmvt_fol, 'surf', '{}.{}.ply'.format(hemi, surf))
        if op.isfile(ply_surf_fname):
            verts[hemi], faces[hemi] = read_ply_file(ply_surf_fname)
            continue
        # pial_npz_fname = op.join(user_fol, 'surf', '{}.{}.npz'.format(hemi, surf))
        # if op.isfile(pial_npz_fname):
        #     d = np.load(pial_npz_fname)
        #     verts[hemi] = d['verts']
        #     faces[hemi] = d['faces']
    if return_faces:
        return verts, faces
    else:
        return verts


def read_ply_file(ply_file):
    if op.isfile(ply_file):
        with open(ply_file, 'r') as f:
            lines = f.readlines()
            verts_num = int(lines[2].split(' ')[-1])
            faces_num = int(lines[6].split(' ')[-1])
            verts_lines = lines[9:9 + verts_num]
            faces_lines = lines[9 + verts_num:]
            verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
            faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
    else:
        print('Can\'t find {}!'.format(ply_file))
        return None, None
    return verts, faces


def apply_trans(trans, points):
    if len(points) == 0:
        return []
    if isinstance(points, list):
        points = np.array(points)
    ndim = points.ndim
    if ndim == 1:
        points = np.array([points])
    points = np.hstack((points, np.ones((len(points), 1))))
    points = np.dot(trans, points.T).T
    points = points[:, :3]
    if ndim == 1:
        points = points[0]
    return points


def time_to_go(now, run, runs_num, runs_num_to_print=10, thread=-1):
    if run % runs_num_to_print == 0 and run != 0:
        time_took = time.time() - now
        more_time = time_took / run * (runs_num - run)
        if thread > 0:
            print('{}: {}/{}, {:.2f}s, {:.2f}s to go!'.format(thread, run, runs_num, time_took, more_time))
        else:
            print('{}/{}, {:.2f}s, {:.2f}s to go!'.format(run, runs_num, time_took, more_time))


def flat_list_of_lists(l):
    return sum(l, [])


def all_items_equall(arr):
    return all([x == arr[0] for x in arr])


def queue_get(queue):
    try:
        if queue is None:
            return None
        else:
            return queue.get(block=False)
    except Empty:
        return None


def calc_normals(vertices, faces):
    # https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    n = normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:,0]] += n
    norm[faces[:,1]] += n
    norm[faces[:,2]] += n
    norm = normalize_v3(norm)
    return norm


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def point_in_mesh(point, closest_vert, closeset_vert_normal, sigma=0, sigma_in=None):
    # https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    p2 = point - closest_vert
    v = p2.dot(closeset_vert_normal) + sigma
    return not(v < 0.0)


def get_brain_surface(subject_fol):
    brain_verts = None
    files = glob.glob(op.join(subject_fol, '**', '*brain_surface*'), recursive=True)
    if len(files) == 1:
        brain_verts, _ = nib.freesurfer.read_geometry(files[0])
    return brain_verts


def get_dural_surface(subject_mmvt_fol, subject_fs_fol='', do_calc_normals=False):
    verts, faces, norms = {}, {}, {}
    # for hemi_ind, hemi in enumerate(['rh', 'lh']):
    verts, faces = read_surf_verts(subject_mmvt_fol, subject_fs_fol, surf='dural', return_faces=True)
        # surf_fname = op.join(subject_fol, 'surf', '{}.dural'.format(hemi))
        # if op.isfile(surf_fname):
        #     # verts[hemi], faces[hemi] = nib.freesurfer.read_geometry(surf_fname)
    if verts['rh'] is None or verts['lh'] is None:
        print("Couldn't find the dural surface!")
        return (None, None, None) if do_calc_normals else (None, None)
    if do_calc_normals:
        for hemi in ['rh', 'lh']:
            norms[hemi] = calc_normals(verts[hemi], faces[hemi])
    if do_calc_normals:
        return verts, faces, norms
    else:
        return verts, faces


def in_shape(xyz, shape):
    x, y, z = xyz
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ang1 = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    ang2 = 2 * np.pi - ang1
    return min(ang1, ang2)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def binary_erosion(ct_data, threshold):
    from scipy import ndimage

    supthresh_locs = np.where(ct_data > threshold)
    ecs = np.zeros(ct_data.shape)
    ecs[supthresh_locs] = 1
    cte = ndimage.binary_erosion(ecs)
    new_ct_data = np.zeros(ct_data.shape)
    new_ct_data[np.where(cte)] = ct_data[np.where(cte)]
    return new_ct_data


def calc_points_length(points):
    return sum([np.linalg.norm(pt2-pt1) for pt1, pt2 in zip(points[:-1], points[1:])])

##### Not in used anymore #########

# from sklearn import mixture
# from sklearn.cluster import KMeans

# def clustering(data, ct_data, n_components, get_centroids=True, clustering_method='knn', output_fol='', threshold=0,
#                covariance_type='full'):
#     if clustering_method == 'gmm':
#         centroids, Y = gmm_clustering(data, n_components, covariance_type, threshold, output_fol)
#     elif clustering_method == 'knn':
#         centroids, Y = knn_clustering(data, n_components, output_fol, threshold)
#     if get_centroids:
#         centroids = np.rint(centroids).astype(int)
#         # for ind, centroid in enumerate(centroids):
#         #     centroids[ind] = find_local_maxima_in_ct(ct_data, centroid, threshold)
#     else: # get max CT intensity
#         centroids = np.zeros(centroids.shape, dtype=np.int)
#         labels = np.unique(Y)
#         for ind, label in enumerate(labels):
#             voxels = data[Y == label]
#             centroids[ind] = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
#     # print(np.all([ct_data[tuple(voxel)] > threshold for voxel in centroids]))
#     return centroids, Y
#
#
# def knn_clustering(data, n_components, output_fol='', threshold=0):
#     kmeans = KMeans(n_clusters=n_components, random_state=0)
#     if output_fol != '':
#         output_fname = op.join(output_fol, 'kmeans_model_{}.pkl'.format(int(threshold)))
#         if not op.isfile(output_fname):
#             kmeans.fit(data)
#             print('Saving knn model to {}'.format(output_fname))
#             joblib.dump(kmeans, output_fname, compress=9)
#         else:
#             kmeans = joblib.load(output_fname)
#     else:
#         kmeans.fit(data)
#     Y = kmeans.predict(data)
#     centroids = kmeans.cluster_centers_
#     return centroids, Y
#
#
# def gmm_clustering(data, n_components, covariance_type='full', output_fol='', threshold=0):
#     gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
#     if output_fol != '':
#         output_fname = op.join(output_fol, 'gmm_model_{}.pkl'.format(int(threshold)))
#         if not op.isfile(output_fname):
#             gmm.fit(data)
#             print('Saving gmm model to {}'.format(output_fname))
#             joblib.dump(gmm, output_fname, compress=9)
#         else:
#             gmm = joblib.load(output_fname)
#     else:
#         gmm.fit(data)
#     Y = gmm.predict(data)
#     centroids = gmm.means_
#     return centroids, Y

# def remove_connected_points(electrodes, points_inside_cylinder, cylinder, ct_data, threshold, ct_header, brain_header):
#     points_to_remove = set()
#     for ind, (pt1, pt2) in enumerate(zip(points_inside_cylinder[:-1], points_inside_cylinder[1:])):
#         pair_electrodes = np.vstack((electrodes[pt1], electrodes[pt2]))
#         vox1, vox2 = t1_ras_tkr_to_ct_voxels(pair_electrodes, ct_header, brain_header)
#         if find_path(vox1, vox2, ct_data, threshold) and ind not in points_to_remove and ind + 1 not in points_to_remove:
#             print('connected points: {}-{}'.format(points_inside_cylinder[ind], points_inside_cylinder[ind + 1]))
#             # points_to_remove.add(ind if ct_data[tuple(vox1)] < ct_data[tuple(vox2)] else ind + 1)
#             pair_dist_to_cylinder = np.min(cdist(pair_electrodes, cylinder), axis=1)
#             dist_to_cylinder_ind = np.argmax(pair_dist_to_cylinder)
#             points_to_remove.add(ind if dist_to_cylinder_ind == 0 else ind + 1)
#     points_to_remove = list(points_to_remove)
#     elecs_to_remove = np.array(points_inside_cylinder)[points_to_remove]
#     if len(points_to_remove) > 0:
#         points_inside_cylinder = np.delete(points_inside_cylinder, points_to_remove, axis=0)
#     return points_inside_cylinder, elecs_to_remove


# def find_path(vox1, vox2, ct_data, threshold):
#     from queue import Queue
#
#     from_vox = [min([vox1[k], vox2[k]]) for k in range(3)]
#     to_vox = [max([vox1[k], vox2[k]]) for k in range(3)]
#     vox = from_vox.copy()
#     queue = Queue()
#     found = False
#     voxs = set()
#     debug_string = ''
#     # steps = [np.array(s) for s in product([0, 1], [0, 1], [0, 1]) if s != (0, 0, 0)]
#     steps = np.eye(3).astype(int)
#     while vox is not None and not found:
#         for step in steps:
#             found = np.array_equal(vox, to_vox)
#             if found:
#                 break
#             if np.all(vox + step <= to_vox) and ct_data[tuple(vox + step)] >= threshold:
#                 debug_string += '{} from {} to {} ({})\n'.format(step, vox, vox + step, ct_data[tuple(vox + step)])
#                 if tuple(vox + step) not in voxs:
#                     queue.put(vox + step)
#                     voxs.add(tuple(vox + step))
#         vox = queue_get(queue)
#     if found:
#         print(debug_string)
#     return found


# def apply_trans(trans, points):
#     points = np.hstack((points, np.ones((len(points), 1))))
#     points = np.dot(trans, points.T).T
#     return points[:, :3]

