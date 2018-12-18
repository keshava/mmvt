import os.path as op
import numpy as np
from src.utils import utils
from src.preproc import anatomy as anat

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def get_t1_vertices_data(subject):
    trans_fname = op.join(MMVT_DIR, subject, 't1_trans.npz')
    trans_dict = utils.Bag(np.load(trans_fname))
    ras_tkr2vox = np.linalg.inv(trans_dict.vox2ras_tkr)
    pial_verts = utils.load_surf(subject, MMVT_DIR, SUBJECTS_DIR)
    t1_data, t1_header = anat.get_data_and_header(subject, 'T1.mgz')
    for hemi in utils.HEMIS:
        output_fname = op.join(MMVT_DIR, subject, 'surf', 'T1-{}.npy'.format(hemi))
        verts = pial_verts[hemi]
        t1_surf_hemi = np.zeros((len(verts)))
        hemi_pial_voxels = np.rint(utils.apply_trans(ras_tkr2vox, verts)).astype(int)
        for vert_ind, t1_vox in zip(range(len(verts)), hemi_pial_voxels):
            t1_surf_hemi[vert_ind] = t1_data[tuple(t1_vox)]
        np.save(output_fname, t1_surf_hemi)


if __name__ == '__main__':
    get_t1_vertices_data('emily')
    print('Done!')