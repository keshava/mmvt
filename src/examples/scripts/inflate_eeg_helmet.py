import bpy
import numpy as np
import os.path as op


def run(mmvt):
    eeg_sensors = np.load(op.join(mmvt.utils.get_user_fol(), 'eeg', 'eeg_sensors_positions.npz'))
    eeg_helmet = bpy.data.objects['eeg_helmet']
    for ind, vert in enumerate(eeg_helmet.data.vertices):
        vert.co = eeg_sensors['pos'][ind] + vert.normal * bpy.context.scene.eeg_helmet_inf


bpy.types.Scene.eeg_helmet_inf = bpy.props.FloatProperty(default=0)


def draw(self, context):
    layout = self.layout
    layout.prop(context.scene, 'eeg_helmet_inf', text='inf factor')


