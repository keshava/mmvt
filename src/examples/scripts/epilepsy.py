import bpy
import glob
import os.path as op
import numpy as np
from scripts_panel import ScriptsPanel
from play_panel import GrabFromPlay, GrabToPlay
from coloring_panel import ColoringMakerPanel as coloring_panel
import mne
from tqdm import tqdm


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    pass


def stc_filter_update(self, context):
    coloring_panel.stc = None


def epilepsy_only_zvals_update(self, context):
    init(_mmvt())


def plot_stc_graph():
    stc_fname = get_stc_fname()
    evokes_fname = op.join(_mmvt().utils.get_user_fol(), 'evokes', '{}.fif'.format(bpy.context.scene.epilepsy_windows))
    _mmvt().coloring.plot_max_stc_graph(stc_fname=stc_fname, modality=bpy.context.scene.epilepsy_modalities,
                                        use_abs=False)


def get_colorbar_title():
    return '{} {} {}'.format(
        bpy.context.scene.epilepsy_modalities.upper(), bpy.context.scene.epilepsy_bands,
        bpy.context.scene.epilepsy_windows)


def plot_stc():
    _mmvt().coloring.plot_stc(coloring_panel.stc, bpy.context.scene.frame_current,
             threshold=bpy.context.scene.coloring_lower_threshold, save_image=False)
    _mmvt().set_colorbar_title(get_colorbar_title())


def get_stc_name():
    mu = _mmvt().mmvt_utils
    return mu.namebase(get_stc_fname())[:-3]


def get_stc_fname():
    mu = _mmvt().mmvt_utils
    modality_fol = op.join(mu.get_user_fol(), 'eeg' if bpy.context.scene.epilepsy_modalities == 'eeg' else 'meg')
    suffix = 'zvals-lh' if bpy.context.scene.epilepsy_only_zvals else 'lh'
    template = '{}-epilepsy-{}-{}-{}?{}-{}.stc'.format(
        mu.get_user(), bpy.context.scene.epilepsy_inverse_methods, bpy.context.scene.epilepsy_modalities,
        bpy.context.scene.epilepsy_windows, bpy.context.scene.epilepsy_bands, suffix)
    files = glob.glob(op.join(modality_fol, '**', template), recursive=True)
    if len(files) == 0:
        print('Couldn\'t find the file! ({})'.format(template))
        return ''
    elif len(files) >=2:
        print('More than one file was found, selecting the first ({})'.format(template))
    return files[0]


def get_image_fname(ind):
    mu = _mmvt().mmvt_utils
    fol = mu.make_dir(op.join(mu.get_user_fol(), 'epilepsy-figures', 'figures'))
    return op.join(fol, '{}_{}_{}_{}_{}.jpg'.format(
        bpy.context.scene.epilepsy_modalities.upper(), bpy.context.scene.epilepsy_bands,
        bpy.context.scene.epilepsy_windows, bpy.context.scene.frame_current, ind))


def save_image():
    mu = _mmvt().mmvt_utils
    _mmvt().render.switch_to_object_mode()
    mu.show_only_render(True)
    image_ind = 1
    image_fname = get_image_fname(image_ind)
    while op.isfile(image_fname):
        image_ind += 1
        image_fname = get_image_fname(image_ind)
    print('Image saved in {}'.format(image_fname))
    bpy.context.scene.render.filepath = image_fname
    view3d_context = mu.get_view3d_context()
    bpy.ops.render.opengl(view3d_context, write_still=True)
    if bpy.context.scene.save_views_with_cb:
        _mmvt().render.add_colorbar_to_image(
            image_fname, bpy.context.scene.cb_ticks_num, bpy.context.scene.cb_ticks_font_size)


def select_stc():
    mu = _mmvt().mmvt_utils
    stc_fname = get_stc_fname()
    if not op.isfile(stc_fname):
        print('Can\'t find {}!'.format(stc_fname))
        return
    print('Loading {}'.format(stc_fname))
    # _mmvt().coloring.set_meg_files(mu.namebase(stc_fname)[:-len('-rh')])
    coloring_panel.stc = mne.read_source_estimate(stc_fname)
    stc_name = get_stc_name()
    data_min, data_max, data_len = _mmvt().coloring.calc_stc_minmax(stc_name=stc_name)
    coloring_panel.smooth_map = _mmvt().meg.calc_smooth_mat(coloring_panel.stc)
    _mmvt().set_colorbar_max_min(data_max, data_min, force_update=True)
    _mmvt().set_colorbar_title(get_colorbar_title())
    T = data_len - 1
    bpy.data.scenes['Scene'].frame_preview_start = 0
    bpy.data.scenes['Scene'].frame_preview_end = T
    if bpy.context.scene.frame_current > T:
        bpy.context.scene.frame_current = T


def plot_stc_peak():
    max_vert, bpy.context.scene.frame_current = coloring_panel.stc.get_peak(
        time_as_index=True, vert_as_index=True, mode=bpy.context.scene.meg_peak_mode)
    print(max_vert, bpy.context.scene.frame_current)
    _mmvt().coloring.plot_stc(coloring_panel.stc, bpy.context.scene.frame_current,
             threshold=bpy.context.scene.coloring_lower_threshold, save_image=False)


def plot_stc_over_time():
    from collections import defaultdict
    mmvt, mu = _mmvt(), _mmvt().utils
    stc = coloring_panel.stc
    if stc is None:
        print('No stc was selected!')
        return
    data, valid_verts = {}, defaultdict(list)
    if mmvt.play.get_play_to() > len(stc.times) - 1:
        mmvt.play.set_play_to(len(stc.times) - 1)
    time = np.arange(mmvt.play.get_play_from(), mmvt.play.get_play_to() + 1)
    stc = coloring_panel.smooth_map.apply(stc)
    data['rh'] = np.ones((stc.rh_data.shape[0], 1)) * -1
    data['lh'] = np.ones((stc.lh_data.shape[0], 1)) * -1
    threshold = mmvt.coloring.get_lower_threshold()
    for t in tqdm(time[::-1]):
        for hemi in mu.HEMIS:
            hemi_data = stc.rh_data[:, t] if hemi == 'rh' else stc.lh_data[:, t]
            verts = np.where(hemi_data >= threshold)[0]
            data[hemi][verts, 0] = t

    data = np.concatenate([data['lh'], data['rh']])
    stc = mne.SourceEstimate(data, stc.vertices, 0, 0, subject=mu.get_user())
    mmvt.colorbar.lock_colorbar_values(False)
    data_max, data_min = time[-1], time[0]
    mmvt.colorbar.set_colorbar_max_min(data_max, data_min, force_update=True)
    mmvt.colorbar.set_colorbar_title('MEG')

    mmvt.coloring.clear_colors()
    mmvt.coloring.plot_stc(
        stc, 0, 0, data_max, data_min, use_abs=False, bigger_or_equal=True)
    mmvt.coloring.set_lower_threshold(threshold) # Set threshold to its previous value


def plot_evoked():
    mu = _mmvt().mmvt_utils
    evoked_fname = op.join(mu.get_user_fol(), 'evokes', '{}.fif'.format(bpy.context.scene.epilepsy_windows))
    bad_channels_fname = op.join(mu.get_user_fol(), 'meg', 'bad_channels.pkl')
    bad_channels = ','.join(mu.load(bad_channels_fname)) if op.isfile(bad_channels_fname) else []
    if not op.isfile(evoked_fname):
        print('Couldn\'t find the evoked file! {}'.format(evoked_fname))
        return
    pick_meg, pick_eeg = 1, 1
    ssp_proj, spatial_colors = 0, 1
    window_title = bpy.context.scene.epilepsy_windows
    mu.run_mmvt_func(
        'src.preproc.meg', 'plot_evoked', flags=
        '-s {} --evo_fname "{}" --pick_meg {} --pick_eeg {} '.format(mu.get_user(), evoked_fname, pick_meg, pick_eeg) +
        '--ssp_proj {} --spatial_colors {} --window_title "{}" --channels_to_exclude {}'.format(
            ssp_proj, spatial_colors, window_title, bad_channels))


def draw(self, context):
    layout = self.layout
    modality = bpy.context.scene.epilepsy_modalities.upper()
    layout.prop(context.scene, 'coloring_lower_threshold', text="Threshold")
    layout.prop(context.scene, 'frame_current', text='Set time')
    layout.prop(context.scene, 'epilepsy_modalities', expand=True)
    layout.prop(context.scene, 'epilepsy_bands', 'Band')
    if len(bpy.types.Scene.epilepsy_inverse_methods[1]['items']) > 1:
        layout.prop(context.scene, 'epilepsy_inverse_methods', '')
    layout.prop(context.scene, 'epilepsy_windows', 'Window')
    layout.prop(context.scene, 'epilepsy_only_zvals', 'Only z-vals')
    col = layout.box().column()
    row = col.row(align=True)
    row.operator(PlotMaxSTCGraph.bl_idname, text="Plot max graph ", icon='IPO_ELASTIC')
    row.operator(PlotEvoked.bl_idname, text="Plot evoked ", icon='IPO_ELASTIC')
    layout.operator(SelectSTC.bl_idname, text="Load File", icon='HAND')
    if not coloring_panel.stc is None:
        row = layout.row(align=True)
        row.operator(EpilepsyPlot.bl_idname, text="Plot {}".format(modality), icon='POTATO')
        row.operator(EpilepsyPeakPlot.bl_idname, text="{} peak".format(modality), icon='POTATO')
        col = layout.box().column()
        row = col.row(align=0)
        row.prop(context.scene, "play_from", text="From")
        row.operator(GrabFromPlay.bl_idname, text="", icon='BORDERMOVE')
        row.prop(context.scene, "play_to", text="To")
        row.operator(GrabToPlay.bl_idname, text="", icon='BORDERMOVE')
        col.operator(EpilepsyPlotStcOverTime.bl_idname, text="Plot {} over time".format(modality), icon='FORCE_HARMONIC')
    layout.operator(EpilepsySaveImage.bl_idname, text="Save image ", icon='ROTATE')


class SelectSTC(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_select_stc"
    bl_label = "mmvt epilepsy_select_stc"
    bl_description = 'Select the stc file'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        select_stc()
        return {"FINISHED"}


class EpilepsyPlotStcOverTime(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_plot_stc_over_time"
    bl_label = "mmvt epilepsy_plot_stc_over_time"
    bl_description = 'Plot stc over time'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_stc_over_time()
        return {"FINISHED"}


class PlotMaxSTCGraph(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_plt_max_stc"
    bl_label = "mmvt epilepsy_plt_max_stc"
    bl_description = 'Plot stc graph'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_stc_graph()
        return {"FINISHED"}


class PlotEvoked(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_plt_evoked"
    bl_label = "mmvt epilepsy_plt_evoked"
    bl_description = 'Plot evoked graph'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_evoked()
        return {"FINISHED"}


class EpilepsyPlot(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_plot"
    bl_label = "mmvt epilepsy_plot"
    bl_description = 'Plots source estimates'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_stc()
        return {"FINISHED"}


class EpilepsyPeakPlot(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_plot_peak"
    bl_label = "mmvt epilepsy_plot_peak"
    bl_description = 'Plots source estimates peak'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        plot_stc_peak()
        return {"FINISHED"}


class EpilepsySaveImage(bpy.types.Operator):
    bl_idname = "mmvt.epilepsy_save_image"
    bl_label = "mmvt epilepsy_sage_image"
    bl_description = 'Save image'
    bl_options = {"UNDO"}

    @staticmethod
    def invoke(self, context, event=None):
        save_image()
        return {"FINISHED"}


bpy.types.Scene.epilepsy_only_zvals = bpy.props.BoolProperty(default=True, update=epilepsy_only_zvals_update)


def init(mmvt):
    mu = mmvt.mmvt_utils
    register()
    bands_names = ['amplitude', 'delta', 'theta', 'alpha', 'beta', 'high_gamma', 'gamma']
    user_fol = mu.get_user_fol()
    meg_fol = op.join(user_fol, 'meg', 'zvals' if bpy.context.scene.epilepsy_only_zvals else 'no-zvals')
    eeg_fol = op.join(user_fol, 'eeg', 'zvals' if bpy.context.scene.epilepsy_only_zvals else 'no-zvals')
    suffix = 'zvals-lh' if bpy.context.scene.epilepsy_only_zvals else 'lh'
    stcs_files = glob.glob(op.join(meg_fol, '{}-epilepsy-*-{}.stc'.format(mu.get_user(), suffix))) + \
                 glob.glob(op.join(eeg_fol, '{}-epilepsy-*-{}.stc'.format(mu.get_user(), suffix)))
    windows, inverse_methods, modalities, bands = set(), set(), set(), set()
    for stc_fname in stcs_files:
        stc_name = mu.namebase(stc_fname)[len('{}-epilepsy-'.format(mu.get_user())):-len('-{}'.format(suffix))]
        ind = stc_name.find('-')
        inverse_methods.add(stc_name[:ind])
        stc_name = stc_name[ind + 1:]
        ind = stc_name.find('-')
        modalities.add(stc_name[:ind])
        stc_name = stc_name[ind + 1:]
        for band in bands_names:
            if stc_name.endswith(band):
                bands.add(band)
                stc_name = stc_name[:-(len(band) + 1)]
                break
        windows.add(stc_name)

    # todo: add a checkbox
    for window_fname in glob.glob(op.join(mu.get_user_fol(), 'evokes', '*.fif')):
        windows.add(mu.namebase(window_fname))

    windows_items = sorted([(c, c, '', ind) for ind, c in enumerate(list(windows))])
    bpy.types.Scene.epilepsy_windows = bpy.props.EnumProperty(
        items=windows_items, description="Windows", update=stc_filter_update)
    if len(windows) > 0:
        bpy.context.scene.epilepsy_windows = windows_items[0][0]

    inverse_methods_items = sorted([(c, c, '', ind) for ind, c in enumerate(list(inverse_methods))])
    bpy.types.Scene.epilepsy_inverse_methods = bpy.props.EnumProperty(
        items=inverse_methods_items, description="Inverse Methods", update=stc_filter_update)
    if len(inverse_methods) > 0:
        bpy.context.scene.epilepsy_inverse_methods = list(inverse_methods)[0]

    bands_items, bands_ind = [], 0
    for band in ['amplitude', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']:
        if band in bands:
            bands_items.append((band, band, '', bands_ind))
            bands_ind += 1
    bpy.types.Scene.epilepsy_bands = bpy.props.EnumProperty(
        items=bands_items, description="Bands", update=stc_filter_update)
    if len(bands) > 0:
        bpy.context.scene.epilepsy_bands = bands_items[0][0]

    modalities_items, modalities_ind = [], 0
    for modality in ['eeg', 'meg', 'meeg']:
        if modality in modalities:
            modalities_items.append((modality, modality.upper(), '', modalities_ind))
            modalities_ind += 1
    bpy.types.Scene.epilepsy_modalities = bpy.props.EnumProperty(
        items=modalities_items, description="Modalities", update=stc_filter_update)
    if len(modalities) > 0:
        bpy.context.scene.epilepsy_modalities = modalities_items[0][0]

    coloring_panel.stc = None


def register():
    try:
        bpy.utils.register_class(SelectSTC)
        bpy.utils.register_class(EpilepsyPlot)
        bpy.utils.register_class(EpilepsyPeakPlot)
        bpy.utils.register_class(PlotMaxSTCGraph)
        bpy.utils.register_class(PlotEvoked)
        bpy.utils.register_class(EpilepsyPlotStcOverTime)
        bpy.utils.register_class(EpilepsySaveImage)
    except:
        pass


def unregister():
    try:
        bpy.utils.unregister_class(SelectSTC)
        bpy.utils.unregister_class(EpilepsyPlot)
        bpy.utils.unregister_class(EpilepsyPeakPlot)
        bpy.utils.unregister_class(PlotMaxSTCGraph)
        bpy.utils.unregister_class(PlotEvoked)
        bpy.utils.unregister_class(EpilepsyPlotStcOverTime)
        bpy.utils.unregister_class(EpilepsySaveImage)
    except:
        pass
