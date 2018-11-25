import os
import os.path as op

mmvt_code_fol = os.environ.get('MMVT_CODE', '')
if mmvt_code_fol == '':
    raise Exception('Please set MMVT_CODE env var, or set it here in the script')

os.chdir(mmvt_code_fol)
from src.mmvt_addon.scripts import run_mmvt
mmvt = run_mmvt.run(subject='sample', atlas='dkt', run_in_background=False, debug=False)#, raise_exp=False)

import mne
mne_sample_data_fol = mne.datasets.sample.data_path()
stc_fname = op.join(mne_sample_data_fol, 'MEG', 'sample', 'sample_audvis-meg-rh.stc')
mmvt.show_hide.show_sagital('left')
stc_fname, _ = mmvt.meg.plot_stc(stc_fname, t=10, threshold=3, cb_percentiles=(1,99), save_image=True)

# Wait for the colorbar to be created
import time
time.sleep(10)

# Show the image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
meg_image = mpimg.imread(stc_fname)
plt.axis('off')
plt.imshow(meg_image)
plt.show()