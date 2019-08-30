import os.path as op
import numpy as np
from scipy.spatial.distance import cdist
import glob
from src.preproc import meg
from src.utils import utils
from src.utils import matlab_utils
from src.utils import freesurfer_utils as fu
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')


def read_coordinates(mat_fname, key, coordinate_system='RAS'):
    coo = matlab_utils.load_mat_to_numpy(mat_fname, key)
    if coordinate_system == 'PRI':
        ras_points = np.zeros(coo.shape)
        for ind in range(coo.shape[0]):
            ras_points[ind, 0] = -coo[ind, 1]
            ras_points[ind, 1] = coo[ind, 0]
            ras_points[ind, 2] = coo[ind, 2]
        coo = ras_points.copy()
    return coo * 10


def load_surfaces(subject):
    vertices = {}
    for surface in ['rh', 'lh', 'Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']:
        if surface in utils.HEMIS:
            surf_fname = op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(surface))
        else:
            surf_fname = op.join(MMVT_DIR, subject, 'subcortical', '{}.npz'.format(surface))
        vertices[surface] = np.load(surf_fname)['verts']
    return vertices


def transform_coordinates_ela(from_subject, to_subject, coordinates):
    from src.preproc import ela_morph_electrodes
    ela_morph_electrodes.calc_elas(
        from_subject, to_subject, coordinates, bipolar=False, atlas='laus125', overwrite=False)


def create_cvs_transformation(subject_from, subjects_to, subjects_dir, openmp=1, step=1):
    for subject_to in subjects_to:
        cmd = 'mri_cvs_register --mov {subject_from} --template {subject_to} ' + \
            '--outdir {subjects_dir}/{subject_to}/mri_cvs_register_from_{subject_from} --nocleanup ' + \
            '--openmp {openmp} --step{}'
        cmd = cmd.format(**locals())
        utils.run_command_in_new_thread(cmd, False)


def transform_coordinates(from_subject, to_subject, coordinates):
    electrodes_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes'))

    vertices_labels_lookup = lu.calc_subject_to_subject_vertices_lookup(from_subject, to_subject)
    # np.savez(op.join(electrodes_fol, 'electrodes_positions.npz'), pos=trans_coords, names=range(len(trans_coords)))
    return

    vertices = load_surfaces(subject)
    distances = np.zeros((len(vertices.keys()), len(trans_coords)))
    verts_indices = np.zeros((len(vertices.keys()), len(trans_coords)))
    for ind, (surface, surf_vertices) in enumerate(vertices.items()):
        dists = cdist(trans_coords, surf_vertices)
        verts_indices[ind] = np.argmin(dists, 1)
        distances[ind] = np.min(dists, 1)
    surfaces_inds = distances.argmin(0)
    cortical_vertices, hemi_indices = {}, {}
    for hemi_ind, hemi in enumerate(utils.HEMIS):
        hemi_indices[hemi] = np.where(surfaces_inds == hemi_ind)[0]
        cortical_vertices[hemi] = verts_indices[:, hemi_indices[hemi]][hemi_ind].astype(np.int)
    # save as electrodes to MMVT
    pos, names = [], []
    for hemi in utils.HEMIS:
        for vert_ind, point_num in zip(cortical_vertices[hemi], hemi_indices[hemi]):
            pos.append(vertices[hemi][vert_ind])
            names.append('{}_{}'.format(point_num, hemi))
    np.savez(op.join(electrodes_fol, 'electrodes_positions.npz'), pos=pos, names=names)
    return hemi_indices, cortical_vertices


def calc_fwd_inv(subject, raw_fname, empty_fname, bad_channels_fname, overwrite_inv=False,
                 overwrite_fwd=False):
    # python -m src.preproc.eeg -s nmr00857 -f calc_inverse_operator,make_forward_solution
    #     --overwrite_inv 0 --overwrite_fwd 0 -t epilepsy
    #     --raw_fname  /autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123/5241495_01_raw.fif
    #     --empty_fname /autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123/5241495_roomnoise_raw.fif
    #     --use_empty_room_for_noise_cov 1
    #     --bad_channels EEG061,EEG02,EEG042,MEG0112,MEG0113
    bad_channels = ','.join(matlab_utils.matlab_cell_arrays_to_dict(bad_channels_fname)['label'])
    trans_fname = op.join(MEG_DIR, subject, '{}-trans.fif'.format(subject))
    args = meg.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_inverse_operator,make_forward_solution',
        task='kaggle',
        inv_fname=op.join(MEG_DIR, subject, '{}-meeg-kaggle-inv.fif'.format(subject)),
        fwd_fname=op.join(MEG_DIR, subject, '{}-meeg-kaggle-fwd.fif'.format(subject)),
        fwd_usingEEG=True,
        overwrite_inv=overwrite_inv,
        overwrite_fwd=overwrite_fwd,
        use_empty_room_for_noise_cov=True,
        bad_channels=bad_channels,
        raw_fname=raw_fname,
        empty_fname=empty_fname,
        cor_fname=trans_fname
    ))
    meg.call_main(args)



if __name__ == '__main__':
    subject = 'wake1'
    template = 'ohad20'
    empty_fname = op.join(MEG_DIR, subject, 'EmptyRoom', '091208_sss.fif')
    mat_fname = op.join(SUBJECTS_DIR, template, '131pntHull.mat')
    bad_channels_fname = op.join(MEG_DIR, subject, 'bad_channels.mat')
    run_num = str(1).zfill(2)

    subjects_to = [utils.namebase(f) for f in  glob.glob(op.join(SUBJECTS_DIR, 'wake*'))]
    create_cvs_transformation(template, subjects_to, SUBJECTS_DIR, openmp=1, step=2)
    # coordinates = read_coordinates(mat_fname, 'newPosBalls', 'PRI')
    # transform_coordinates(template, subject, coordinates)

    raw_files = glob.glob(op.join(MEG_DIR, subject, 'run_*_sss.fif'))
    # for raw_fname in raw_files:
    #     calc_fwd_inv(subject, raw_fname, empty_fname, bad_channels_fname, overwrite_inv=False, overwrite_fwd=False)
    print('sdf')