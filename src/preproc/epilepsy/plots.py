

def plot_topomaps(subject, modality, windows, bad_channels, parallel=True):
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'topomaps'))
    params = [(subject, modality, window_fname, bad_channels, figs_fol) for window_fname in windows]
    utils.run_parallel(_plot_topomaps_parallel, params, len(windows) if parallel else 1)


def _plot_topomaps_parallel(p):
    subject, modality, window_fname, bad_channels, figs_fol = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    evo_fname = op.join(MMVT_DIR, subject, 'evoked', '{}.fif'.format(window))
    if not op.isfile(evo_fname):
        utils.make_link(window_fname, evo_fname)
    fig_fname = op.join(figs_fol, '{}.jpg'.format(window))
    if op.isfile(fig_fname):
        return
    if bad_channels != 'bads':
        bad_channels = bad_channels.split(',')
    module.plot_topomap(
        subject, evo_fname, times=[0], find_peaks=True, same_peaks=False, n_peaks=5, bad_channels=bad_channels,
        title=window, save_fig=True, fig_fname=fig_fname)


def plot_evokes(subject, modality, windows, bad_channels, parallel=True, overwrite=False):
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'evokes'))
    params = [(subject, modality, window_fname, bad_channels, figs_fol, overwrite) for window_fname in windows]
    utils.run_parallel(_plot_evokes_parallel, params, len(windows) if parallel else 1)


def _plot_evokes_parallel(p):
    subject, modality, window_fname, bad_channels, figs_fol, overwrite = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    evo_fname = op.join(MMVT_DIR, subject, 'evoked', '{}.fif'.format(window))
    if not op.isfile(evo_fname):
        utils.make_link(window_fname, evo_fname)
    fig_fname = op.join(figs_fol, '{}.jpg'.format(window))
    if op.isfile(fig_fname) and not overwrite:
        return
    if bad_channels != 'bads':
        bad_channels = bad_channels.split(',')
    module.plot_evoked(
        subject, evo_fname, window_title=window, exclude=bad_channels, save_fig=True,
        fig_fname=fig_fname, overwrite=overwrite)


def plot_modalities_power_spectrums_with_graph(
        subject, modalities, window_fname, figure_name='', percentiles=[5, 95], inverse_method='dSPM', ylims=[-18, 6],
        file_type='jpg', cb_ticks = [], cb_ticks_font_size=12, figure_fol=''):

    evoked = mne.read_evokeds(window_fname)[0]
    times = evoked.times if len(evoked.times) % 2 == 0 else evoked.times[:-1]
    times = utils.downsample(times, 2)
    min_t, max_t = round(times[0]), round(times[-1])
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, 125, 5)])
    # bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    bands = dict(delta=[1, 4], high_gamma=[65, 120])
    if figure_fol == '':
        figure_fol = op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum')

    fig, ax = plt.subplots(3, len(modalities), figsize=(20, 10))
    for ind, modality in enumerate(modalities):
        powers_ax = ax[0, ind]
        powers_ax.set_title(modality.upper(), fontdict={'fontsize': 18})
        positive_powers_ax = ax[1, ind]
        negative_powers_ax = ax[2, ind]
        root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
            subject, inverse_method, modality, '{window}'))
        window = utils.namebase(window_fname)
        window_output_fname = output_fname.format(window=window)
        d = np.load(window_output_fname)
        powers_negative, powers_positive = calc_masked_negative_and_positive_powers(d['min'], d['max'], percentiles)

        im1 = _plot_powers(powers_negative, powers_ax, times)
        im2 = _plot_powers(powers_positive, powers_ax, times)
        if ind == 0:
            powers_ax.set_ylabel('Frequency (Hz)')
        else:
            powers_ax.set_yticks([])
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(powers_ax, width="5%", height="100%", loc=5,
                               bbox_to_anchor=(1.15, 0, 1, 1), bbox_transform=powers_ax.transAxes)
            cb = plt.colorbar(im2, cax=axins)
            if cb_ticks != []:
                cb.set_ticks(cb_ticks)
            cb.ax.tick_params(labelsize=cb_ticks_font_size)
            cb.ax.set_ylabel('dBHZ Z-Score', color='black', fontsize=cb_ticks_font_size)

        # for powers, axis, positive in zip([powers_positive, powers_negative], ax[1:2, ind], [True, False]):
        #     for band_name, band_freqs in bands.items():
        #         idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        #         band_power = np.mean(powers[idx, :], axis=0)
        #         band_power[abs(band_power) < 2] = 0
        #         axis.plot(times, band_power.T, label=band_name)
        #     axis.set_xlim([min_t, max_t])
        #     axis.set_ylim([2, ylims[1]] if positive else [ylims[0], -2])
        #     if ind == 0:
        #         axis.set_ylabel('Positive Z-Scores' if positive else 'Negative Z-Scores')
        #     else:
        #         axis.set_yticks([])
        #         axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # negative_powers_ax.set_xlabel('Time (s)')

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_negative[idx, :], axis=0)
            band_power[abs(band_power) < 2] = 0
            positive_powers_ax.plot(times, band_power.T, label=band_name.replace('_', ' '))
        positive_powers_ax.set_xlim([min_t, max_t])
        positive_powers_ax.set_ylim([2, ylims[1]])
        if ind == 0:
            positive_powers_ax.set_ylabel('Positive Z-Scores')
        else:
            positive_powers_ax.set_yticks([])
        if ind == len(modalities) - 1:
            positive_powers_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_positive[idx, :], axis=0)
            band_power[abs(band_power) < 2] = 0
            negative_powers_ax.plot(times, band_power.T, label=band_name.replace('_', ' '))
        negative_powers_ax.set_xlim([min_t, max_t])
        negative_powers_ax.set_ylim([ylims[0], -2])
        if ind == 0:
            negative_powers_ax.set_ylabel('Negative Z-Scores')
        else:
            negative_powers_ax.set_yticks([])
        if ind == len(modalities) - 1:
            negative_powers_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        negative_powers_ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=0.92, top=None, wspace=None, hspace=None)
    if figure_name != '':
        plt.savefig(op.join(figure_fol, '{}.{}'.format(figure_name, file_type)), dpi=300)
        plt.close()
    else:
        plt.show()

