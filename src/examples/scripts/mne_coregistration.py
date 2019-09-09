import os


def run(mmvt):
    mu = mmvt.utils
    os.chdir(mu.get_mmvt_code_root())
    cmd = 'python -m src.preproc.anatomy -s {} -f mne_coregistration'.format(mu.get_user())
    print(cmd)
    mu.run_command_in_new_thread(cmd, False)
    #              mu.add_mmvt_code_root_to_path()
    #             from src.preproc import fMRI
    #             importlib.reload(fMRI)
    # mne.gui.coregistration()
