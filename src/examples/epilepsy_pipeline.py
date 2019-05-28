from src.preproc import eeg
from src.preproc import meg
import glob


def calc_eeg_induced_power(subject, windows_fnames):
    for window_fname in windows_fnames:
        eeg_args = eeg.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            task='epilepsy',
            calc_source_band_induced_power=True,
            evo_fname=window_fname,
            inv_fname='nmr00857-epilepsy-eeg-inv.fif',
            n_jobs=1,
            overwrite_stc=False
        ))
        eeg.call_main(eeg_args)


def calc_meg_induced_power(subject, windows_fnames):
    for window_fname in windows_fnames:
        meg_args = meg.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            task='epilepsy',
            calc_source_band_induced_power=True,
            evo_fname=window_fname,
            inv_fname='nmr00857-epilepsy-meg-inv',
            fwd_usingEEG=False,
            n_jobs=1,
            overwrite_stc=False
        ))
        meg.call_main(meg_args)
        del meg_args


if __name__ == '__main__':
    windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    # calc_eeg_induced_power('nmr00857', windows)
    calc_meg_induced_power('nmr00857', windows)