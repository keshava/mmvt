% https://github.com/mne-tools/mne-matlab
% http://old.fieldtriptoolbox.org/development/integrate_with_mne
addpath('/home/npeled/fieldtrip/fieldtrip-20190111');
addpath('/home/npeled/MNE/share/matlab');
ft_defaults

global FIFF;
FIFF = fiff_define_constants();

SUBJECT_MEG_ROOT = '/home/npeled/meg/MSIT/hc016/';
epochs_fname = [SUBJECT_MEG_ROOT 'hc016_MSIT_Onset-epo.fif'];

% First, we baseline correct the data that we are going to use for the TFRs - i.e. demean the epochs.
cfg = [];
cfg.demean          = 'yes';
cfg.baselinewindow  = 'all';
cfg.dataset = epochs_fname;
epochs_prep = ft_preprocessing(cfg);

bands = [[4, 8]; [8, 15]; [15, 30]; [30, 55]; [65, 200]];
bands_psd = zeros(length(bands), 306);
for band_ind=1:length(bands)
    % Get PSD with multitapers
    cfg = [];
    cfg.output      = 'pow';          % Return PSD
    cfg.channel     = {'MEG'};  % Calculate for MEG ({'MEG','EEG'} and EEG)
    cfg.method      = 'mtmfft';
    cfg.taper       = 'dpss';         % Multitapers based on Slepian sequences
    cfg.tapsmofrq   = 2;              % Smoothing +/- 2 Hz
    cfg.pad         = 'nextpow2';
    cfg.foilim      = [bands(band_ind, 1), bands(band_ind, 2)];
    psd_dpss        = ft_freqanalysis(cfg, epochs_prep);
    bands_psd(band_ind, :) = mean(psd_dpss.powspctrm, 2);
end
save([SUBJECT_MEG_ROOT, 'hc016_MSIT_Onset-psd.mat'], 'bands_psd');

