import mne
import os.path as op
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

data_path = sample.data_path()
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'


def mne_python():
    subjects_dir = data_path + '/subjects'

    # Read data
    evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                              baseline=(None, 0))
    fwd = mne.read_forward_solution(fname_fwd)
    cov = mne.read_cov(fname_cov)

    inv = make_inverse_operator(evoked.info, fwd, cov, loose=0., depth=0.8,
                                verbose=True)

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    kwargs = dict(initial_time=0.08, hemi='both', subjects_dir=subjects_dir,
                  size=(600, 600))

    stc = abs(apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True))
    stc = mne.SourceEstimate(stc.data * 1e10, stc.vertices, stc.tmin , stc.tstep, subject='sample')
    stc.save(data_path + '/MEG/sample/sample_audvis_MNE')
    # brain = stc.plot(figure=1, **kwargs)
    # brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)


def mmvt():
    from src.preproc import meg
    # Call the MMVT MEG preprocessing to create the inverse solution and the source estimate
    args = meg.read_cmd_args(dict(
        subject='sample', task='audvis', inverse_method='MNE', pick_ori=None,
        function='calc_inverse_operator,calc_stc', inv_loose=0,
        evo_fname=fname_evoked, fwd_fname=fname_fwd, noise_cov_fname=fname_cov))
    meg.call_main(args)
    stc_fname = meg.get_stc_fname(args).format(hemi='lh')
    t = meg.time_to_index(0.08)
    stc_fname, _ = mmvt.meg.plot_stc(stc_fname, t=168, threshold=1.43, save_image=True)


if __name__ == '__main__':
    mne_python()