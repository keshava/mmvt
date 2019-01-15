% https://github.com/mne-tools/mne-matlab
% http://old.fieldtriptoolbox.org/development/integrate_with_mne
%addpath('/home/npeled/fieldtrip/fieldtrip-20190111');
addpath('/autofs/space/thibault_001/users/npeled/fieldtrip/fieldtrip-20190114');
%addpath('/home/npeled/MNE/share/matlab');
ft_defaults

%global FIFF;
%FIFF = fiff_define_constants();

SUBJECT_MEG_ROOT = '/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/epochs/';
SUBJECTS = {'ep002', 'ep003', 'ep004', 'ep005', 'ep006', 'ep007', 'ep008', 'ep009', 'ep010', 'ep011', 'ep012', 'ep016', 'hc001', 'hc002', 'hc003', 'hc004', 'hc005', 'hc006', 'hc007', 'hc008', 'hc009', 'hc010', 'hc011', 'hc012', 'hc013', 'hc014', 'hc015', 'hc016', 'hc017', 'hc019', 'hc020', 'hc021', 'hc022', 'hc023', 'hc024', 'hc025', 'hc026', 'hc027', 'hc028', 'hc029', 'hc030', 'hc031', 'hc032', 'hc033', 'hc034', 'hc035', 'hc036', 'hc042', 'hc044', 'pp001', 'pp002', 'pp003', 'pp004', 'pp005', 'pp006', 'pp007', 'pp008', 'pp009', 'pp010', 'pp011', 'pp013', 'pp014', 'pp015', 'pp016', 'sp004'};
BANDS = [[4, 8]; [8, 15]; [15, 30]; [30, 55]; [65, 200]];
TASKS = {'MSIT', 'ECR'};

for subject_ind = 1:length(SUBJECTS)
    subject = SUBJECTS{subject_ind};
    for task_ind = 1:length(TASKS)
        task = TASKS{task_ind};
        epochs_fname = [SUBJECT_MEG_ROOT, subject, '/', subject, '_', task, '_meg_Onset_ar-epo.fif'];

        % First, we baseline correct the data that we are going to use for the TFRs - i.e. demean the epochs.
        cfg = [];
        cfg.demean          = 'yes';
        cfg.baselinewindow  = 'all';
        cfg.dataset = epochs_fname;
        epochs_prep = ft_preprocessing(cfg);

        bands_psd = zeros(length(BANDS), 306);
        for band_ind=1:length(BANDS)
            % Get PSD with multitapers
            cfg = [];
            cfg.output      = 'pow';          % Return PSD
            cfg.channel     = {'MEG'};  % Calculate for MEG ({'MEG','EEG'} and EEG)
            cfg.method      = 'mtmfft';
            cfg.taper       = 'dpss';         % Multitapers based on Slepian sequences
            cfg.tapsmofrq   = 2;              % Smoothing +/- 2 Hz
            cfg.pad         = 'nextpow2';
            cfg.foilim      = [BANDS(band_ind, 1), BANDS(band_ind, 2)];
            psd_dpss        = ft_freqanalysis(cfg, epochs_prep);
            bands_psd(band_ind, :) = mean(psd_dpss.powspctrm, 2);
        end
        save([SUBJECT_MEG_ROOT, subject, '/', subject, '_', task, '_Onset-psd.mat'], 'bands_psd');
    end
end
