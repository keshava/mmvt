import bpy
import numpy as np
import os.path as op
import time
import mmvt_utils as mu
import os
import glob
from pprint import pprint

PARENT_OBJ = 'dti'
TRACULA = 'tracula'
SUBJECT_DTI_FOL = op.join(mu.get_user_fol(), 'dti')

TRACULA_PATHWAYS_DIC = {
    'lh.cst_AS': 'Left corticospinal tract',
    'rh.cst_AS': 'Right corticospinal tract',
    'lh.ilf_AS': 'Left inferior longitudinal fasciculus',
    'rh.ilf_AS': 'Right inferior longitudinal fasciculus',
    'lh.unc_AS': 'Left uncinate fasciculus',
    'rh.unc_AS': 'Right uncinate fasciculus',
    'fmajor_PP': 'Corpus callosum - forceps major',
    'fminor_PP': 'Corpus callosum - forceps minor',
    'lh.atr_PP': 'Left anterior thalamic radiations',
    'rh.atr_PP': 'Right anterior thalamic radiations',
    'lh.ccg_PP': 'Left cingulum - cingulate gyrus endings',
    'rh.ccg_PP': 'Right cingulum - cingulate gyrus endings',
    'lh.cab_PP': 'Left cingulum - angular bundle',
    'rh.cab_PP': 'Right cingulum - angular bundle',
    'lh.slfp_PP': 'Left superior longitudinal fasciculus - parietal endings',
    'rh.slfp_PP': 'Right superior longitudinal fasciculus - parietal endings',
    'lh.slft_PP': 'Left superior longitudinal fasciculus - temporal endings',
    'rh.slft_PP': 'Right superior longitudinal fasciculus - temporal endings'
}

TRACULA_POSTFIX = '_avg33_mni_bbr'


def set_dti_pathways(self=None, value=None):
    if bpy.context.scene.dti_type == TRACULA:
        bpy.types.Scene.dti_pathways = bpy.props.EnumProperty(items=get_tracula_pathways(), description="pathways")


def get_tracula_pathways():
    pathways = [get_group_name(pkl_fname) for pkl_fname in glob.glob(op.join(SUBJECT_DTI_FOL, TRACULA, '*.pkl'))]
    items = [(pathway, TRACULA_PATHWAYS_DIC[pathway], '', ind) for ind, pathway in enumerate(pathways)]
    items = sorted(items, key=lambda x:x[1])
    return items


def get_tracula_traks():
    tracks = glob.glob(op.join(SUBJECT_DTI_FOL, '*_tracks.npy'))
    items, ind = [], 0
    for track_fname in tracks:
        track_name = mu.namebase(track_fname)[:-len('_tracks')]
        header_fname = op.join(SUBJECT_DTI_FOL, '{}_header.pkl'.format(track_name))
        print(track_name, header_fname)
        if op.isfile(header_fname):
            items.append((track_name, track_name, '', ind))
    items = sorted(items)
    return items


def plot_tracks():
    tracks_name = bpy.context.scene.dti_tracks
    tracks_fname = op.join(SUBJECT_DTI_FOL, '{}_tracks.npy'.format(tracks_name))
    tracks_header = op.join(SUBJECT_DTI_FOL, '{}_header.pkl'.format(tracks_name))
    world_matrix = mu.get_matrix_world()
    tracks = np.load(tracks_fname) * 0.1
    header = mu.load(tracks_header)

    layers_dti = [False] * 20
    dti_layer = DTIPanel.addon.CONNECTIONS_LAYER
    layers_dti[dti_layer] = True
    mu.create_empty_if_doesnt_exists(tracks_name, DTIPanel.addon.CONNECTIONS_LAYER, None, PARENT_OBJ)
    parent_obj = bpy.data.objects[tracks_name]

    # N = len(tracks)
    # now = time.time()
    # for ind, track in enumerate(tracks):
    #     mu.time_to_go(now, ind, N, 100)
    cur_obj = mu.create_spline(tracks, layers_dti, bevel_depth=0.01)
    cur_obj.name = tracks_name
    cur_obj.parent = parent_obj


def get_group_name(pkl_fname):
    pathway = mu.namebase(pkl_fname)
    pathway = pathway[:-len(TRACULA_POSTFIX)]
    return pathway


def plot_pathway(self, context, layers_dti, pathway_name, pathway_type):
    if pathway_type == TRACULA:
        pkl_fname = op.join(SUBJECT_DTI_FOL, TRACULA, '{}{}.pkl'.format(pathway_name, TRACULA_POSTFIX))
        mu.create_empty_if_doesnt_exists(pathway_name, DTIPanel.addon.CONNECTIONS_LAYER, None, PARENT_OBJ)
        parent_obj = bpy.data.objects[pathway_name]
        tracks = mu.load(pkl_fname)
        N = len(tracks)
        now = time.time()
        for ind, track in enumerate(tracks[:1000]):
            mu.time_to_go(now, ind, N, 100)
            track = track * 0.1
            # pprint(track)
            cur_obj = mu.create_spline(track, layers_dti, bevel_depth=0.01)
            # cur_obj.scale = [0.1] * 3
            cur_obj.name = '{}_{}'.format(pathway_name, ind)
            cur_obj.parent = parent_obj


class PlotTracks(bpy.types.Operator):
    bl_idname = "mmvt.plot_tracks"
    bl_label = "mmvt plt tracks"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_tracks()


class PlotPathway(bpy.types.Operator):
    bl_idname = "mmvt.plot_pathway"
    bl_label = "mmvt plt pathway"
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        if not bpy.data.objects.get(PARENT_OBJ):
            self.report({'ERROR'}, 'No parent node was found, you first need to create the connections.')
        else:
            pathway_type = bpy.context.scene.dti_type
            pathway_name = bpy.context.scene.dti_pathways
            layers_dti = [False] * 20
            dti_layer = DTIPanel.addon.CONNECTIONS_LAYER
            layers_dti[dti_layer] = True
            plot_pathway(self, context, layers_dti, pathway_name, pathway_type)
        return {"FINISHED"}


def dti_draw(self, context):
    layout = self.layout
    # layout.prop(context.scene, "dti_type", text="")
    if len(DTIPanel.pathways) > 0:
        layout.prop(context.scene, "dti_pathways", text="")
        layout.operator(PlotPathway.bl_idname, text="plot pathway", icon='POTATO')
    if len(DTIPanel.tracks) > 0:
        layout.prop(context.scene, "dti_tracks", text="")
        layout.operator(PlotTracks.bl_idname, text="plot tracks", icon='POTATO')


# bpy.types.Scene.dti_type = bpy.props.EnumProperty(items=[(TRACULA, TRACULA, "", 1)], description="DTI source",
#                                                   set=set_dti_pathways)
bpy.types.Scene.dti_pathways = bpy.props.EnumProperty(items=get_tracula_pathways(), description="pathways")
bpy.types.Scene.dti_tracks = bpy.props.EnumProperty(items=get_tracula_traks(), description="traks")


class DTIPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "DTI"
    addon = None
    init = False
    tracks = []
    pathways = []
    # d = mu.Bag({})

    def draw(self, context):
        dti_draw(self, context)


def check_for_dti_files():
    # Check if the dti files exist
    # pathway_types = [TRACULA]
    # for pathway_type in pathway_types:
    #     if pathway_type == TRACULA:
    DTIPanel.pathways = [get_group_name(pkl_fname) for pkl_fname in glob.glob(op.join(SUBJECT_DTI_FOL, TRACULA, '*.pkl'))]
    DTIPanel.tracks = glob.glob(op.join(SUBJECT_DTI_FOL, '*_tracks.npy'))
    return len(DTIPanel.pathways) > 0 or len(DTIPanel.tracks) > 0


def init(addon):
    if not check_for_dti_files():
        unregister()
        DTIPanel.init = False
    else:
        register()
        DTIPanel.addon = addon
        mu.create_empty_if_doesnt_exists(PARENT_OBJ, addon.BRAIN_EMPTY_LAYER, None, 'Brain')
        DTIPanel.init = True
        # DTIPanel.d = d
        # print('DRI panel initialization completed successfully!')


def register():
    try:
        unregister()
        bpy.utils.register_class(DTIPanel)
        bpy.utils.register_class(PlotPathway)
        bpy.utils.register_class(PlotTracks)
        # print('DTI Panel was registered!')
    except:
        print("Can't register DTI Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(DTIPanel)
        bpy.utils.unregister_class(PlotPathway)
        bpy.utils.unregister_class(PlotTracks)
    except:
        pass
        # print("Can't unregister DTI Panel!")

