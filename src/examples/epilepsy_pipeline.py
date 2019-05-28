from src.preproc import eeg
from src.preproc import meg
import glob


def calc_induced_power(subject, windows_fnames):

    for window_fname in windows_fnames:
        # EEG
        args = eeg.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            calc_source_band_induced_power=True,
            evo_fname=window_fname,
            n_jobs=1,
            overwrite_stc=False
        ))
        eeg.call_main(args)
        # MEG
        args = meg.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            calc_source_band_induced_power=True,
            evo_fname=window_fname,
            fwd_usingEEG=False,
            n_jobs=1,
            overwrite_stc=False
        ))
        eeg.call_main(args)


if __name__ == '__main__':
    windows = glob.glob('/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/EPI_interictal/*.fif')
    calc_induced_power('nmr00857', windows)