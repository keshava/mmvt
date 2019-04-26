import os.path as op
import numpy as np
import bpy


def run(mmvt):
    mu = mmvt.mmvt_utils
    x = mu.load(op.join(mu.get_mmvt_root(), 'colin27', 'electrodes', 'electrodes_data.pkl'))
    d = x[0]['electrodes']
    first_key = list(x[0]['electrodes'].keys())[0]
    T = len(d[first_key])
    conditions = ['interference', 'noninterference']
    from_electrodes = ['LOF1', 'RAF5']
    to_electrodes = ['LPF2-LPF1', 'LAF2-LAF1']
    parent_obj = bpy.data.objects.get('Deep_electrodes')
    all_data = []
    if parent_obj is None:
        print('Deep_electrodes is None!')
        return
    parent_obj.animation_data_clear()
    for elec_ind, (from_electrode, to_electrode) in enumerate(zip(from_electrodes, to_electrodes)):
        cur_obj = bpy.data.objects.get(to_electrode, None)
        if cur_obj is None:
            print('{} is None!'.format(to_electrode))
            continue
        data = np.zeros((T, 2))
        for k in range(len(conditions)):
            data[:, k] = np.array(d['{}_{}'.format(from_electrode, conditions[k])])
        all_data.append(data)
        mmvt.data.add_data_to_electrode(data, cur_obj, to_electrode, conditions, T)

        parent_data = np.diff(data, axis=1)
        mu.insert_keyframe_to_custom_prop(parent_obj, to_electrode, 0, 1)
        mu.insert_keyframe_to_custom_prop(parent_obj, to_electrode, 0, T + 2)

        for ind in range(T):
            mu.insert_keyframe_to_custom_prop(parent_obj, to_electrode, parent_data[ind], ind + 2)

        fcurves = parent_obj.animation_data.action.fcurves[elec_ind]
        mod = fcurves.modifiers.new(type='LIMITS')
    output_fname = op.join(mu.get_user_fol(), 'electrodes', 'electrodes_bipolar_data.npz')
    np.savez(output_fname, data=np.array(all_data), names=to_electrodes, conditions=conditions, colors=[])