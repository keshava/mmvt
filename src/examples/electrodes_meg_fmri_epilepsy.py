import os.path as op
from src.preproc import meg, eeg
from src.utils import utils
import mne
import glob
import numpy as np
from matplotlib import pyplot as plt


LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')

MEG_ROOT = '/autofs/space/karima_002/users/Machine_Learning_Clinical_MEG_EEG_Resting/raw_preprocessed'


def analyze_meg(seizure_time, seizure_len):
    subject = 'nmr01209'
    raw, evoked = None, None
    meg_raw_fnames = [op.join(MEG_ROOT, 'nmr01209_6213848_07', 'nmr01209_6213848_07_Resting_eeg_meg_ica-raw.fif'),
                  op.join(MEG_DIR, subject, 'nmr01209_6213848_07_Resting_eeg_meg_ica-raw.fif')]
    meg_raw_fname = [f for f in meg_raw_fnames if op.isfile(f)][0]
    # empty_room_fname = '/space/megraid/77/MEG/noise/no_name/'
    meg_raw_fname_seizure = op.join(MEG_DIR, subject, 'meg', '{}_meg_seizure-raw.fif'.format(subject))
    meg_evoked_fname = op.join(MEG_DIR, subject, 'meg', '{}_meg_seizure-ave.fif'.format(subject))
    eeg_evoked_fname = op.join(MEG_DIR, subject, 'meg', '{}_eeg_seizure-ave.fif'.format(subject))

    electrodes_all_data_fname = op.join(MMVT_DIR, subject, 'electrodes', 'all', 'electrodes_data_all.npy')
    electrodes_all_meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'all', 'electrodes_all_meta_data.npz')
    electrodes_data_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_data.npy')
    electrodes_meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_meta_data.npz')
    electrodes_from_t, electrodes_to_t = 630000, 645000

    if not op.isfile(meg_raw_fname_seizure):
        raw = mne.io.read_raw_fif(meg_raw_fname)
        raw.set_eeg_reference('average', projection=True)  # set EEG average reference
        raw = raw.crop(seizure_time-2, seizure_time+seizure_len)
        raw.save(meg_raw_fname_seizure)
        # raw.plot(block=True)
        # raw.plot(butterfly=True, group_by='position')
        meg.read_sensors_layout(subject, info=raw.info, overwrite_sensors=False)
        eeg.read_sensors_layout(subject, info=raw.info, overwrite_sensors=False)

    if not op.isfile(meg_evoked_fname) or not op.isfile(eeg_evoked_fname):
        if raw is None:
            raw = mne.io.read_raw_fif(meg_raw_fname_seizure)
        evoked = mne.EvokedArray(raw.get_data(), raw.info, comment='seizure')
        eeg_evoked = evoked.pick_types(meg=False, eeg=True)
        mne.write_evokeds(eeg_evoked_fname, eeg_evoked)
        meg.save_evokes_to_mmvt(eeg_evoked, [1], subject, modality='eeg')

        meg_evoked = evoked.pick_types(meg=True, eeg=False)
        mne.write_evokeds(meg_evoked_fname, meg_evoked)
        meg.save_evokes_to_mmvt(evoked, [1], subject,  modality='meg')

    if not op.isfile(electrodes_data_fname):
        x = np.load(electrodes_all_data_fname)
        x = x[:, electrodes_from_t:electrodes_to_t, :]
        np.save(electrodes_data_fname, x)
        d = utils.Bag(np.load(electrodes_all_meta_fname))
        times = d.times[electrodes_from_t:electrodes_to_t]
        np.savez(electrodes_meta_fname, names=d.names, conditions=d.conditions, times=times)

    meg.create_helmet_mesh(subject, overwrite_faces_verts=True)
    eeg.create_helmet_mesh(subject, overwrite_faces_verts=True)

    overwrite_source_bem = True
    args = meg.read_cmd_args(dict(
        subject=subject,
        function='make_forward_solution,calc_inverse_operator,calc_stc',
        recreate_src_spacing='ico5',
        fwd_recreate_source_space=overwrite_source_bem,
        recreate_bem_solution=overwrite_source_bem,
        contrast='seizure', task='seizure',
        raw_fname=meg_raw_fname_seizure,
        use_empty_room_for_noise_cov=True,
        empty_fname='empty_room_raw.fif',
        overwrite_fwd=True,
        # pick_meg=True, pick_eeg=False,
        # fwd_usingMEG=True, fwd_usingEEG=False,
    ))
    meg.call_main(args)


if __name__ == '__main__':
    # MEG:

    # MRI:
    '/autofs/space/megraid_clinical/MEG-MRI/seder/freesurfer/nmr01209'
    # fMRI:
    '/autofs/space/megraid_clinical/MEG-MRI/clin_6213848/20171127'
    # Reports:
    '/autofs/space/megraid_archive/80/MEG-clin/naoro/report/6213848_171127'

    seizure_time = 176.1 #s
    analyze_meg(seizure_time, 5)