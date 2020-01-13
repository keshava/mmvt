from src.utils import utils
from src.utils import args_utils as au
import os.path as op
import numpy as np

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')

patients=[ 'MG51b','MG72','MG73','MG83','MG76','MG84','MG85','MG86','MG87','MG90','MG91','MG92','MG93','MG94','MG95','MG96','MG98','MG100','MG103','MG104','MG105','MG106','MG107','MG108','MG109','MG110','MG111','MG112','MG114','MG115','MG116','MG118','MG120','MG121','MG122','BW36','BW37','BW38','BW39','BW40','BW42','BW43','BW44']
for pat in patients:
    electrodes_fname = op.join(SUBJECTS_DIR, pat.lower(), 'electrodes', '{}_RAS.csv'.format(pat.lower()))
    if op.isfile(electrodes_fname):
        electrodes = np.genfromtxt(electrodes_fname, dtype=str, delimiter=',', skip_header=1)
        electrodes_num = electrodes.shape[0]
        print('{}'.format(electrodes_num))
    else:
        print('0')
