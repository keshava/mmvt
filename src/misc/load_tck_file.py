import os.path as op
from src.utils import utils
import nibabel as nib
import numpy as np

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
DTI_DIR = utils.get_link_dir(LINKS_DIR, 'dti')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def load_tck_file(subject, tck_file_name):
    input_fname = op.join(DTI_DIR, subject, tck_file_name)
    if not op.isfile(input_fname):
        print('Cannot find the input file! {}!'.format(input_fname))
        return False
    tck_file = nib.streamlines.load(input_fname)
    tracks = tck_file.streamlines.get_data()
    header = tck_file.header
    return tracks, header


def save_to_mmvt(subject, tracks, header, tracks_name):
    dti_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'dti'))
    np.save(op.join(dti_fol, '{}_tracks.npy'.format(tracks_name)), tracks)
    utils.save(header, op.join(dti_fol, '{}_header.pkl'.format(tracks_name)))


def plot_tracks(tracks):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(tracks[:, 0], tracks[:, 1], tracks[:, 2])
    plt.show()


def nipype_convert(input_fname):
    output_fname = utils.replace_file_type(input_fname, 'trk')
    image_fname = utils.replace_file_type(input_fname, 'nii')
    # conda install --channel conda-forge nipype (or pip install nipype)
    # conda install -c conda-forge dipy
    # https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.mrtrix.convert.html
    import nipype.interfaces.mrtrix as mrt
    '''
    Mandatory Inputs:
        in_file (a pathlike object or string representing an existing file):
            The input file for the tracks in MRTrix (.tck) format.
    Optional Inputs
        image_file (a pathlike object or string representing an existing file):
            The image the tracks were generated from.
        matrix_file (a pathlike object or string representing an existing file):
            A transformation matrix to apply to the tracts after they have been generated 
            (from FLIRT - affine transformation from image_file to registration_image_file).
        out_filename (a pathlike object or string representing a file):
            The output filename for the tracks in TrackVis (.trk) format. (Nipype default value: converted.trk)
        registration_image_file (a pathlike object or string representing an existing file):
            The final image the tracks should be registered to.
    '''
    tck2trk = mrt.MRTrix2TrackVis()
    tck2trk.inputs.in_file = input_fname
    # tck2trk.inputs.image_file = nii_fname
    tck2trk.inputs.out_filename = output_fname
    tck2trk.run()
    return op.isfile(output_fname)


if __name__ == '__main__':
    subject = 'Broad_52_SC'
    tck_file_name = 'sub-52_Koganemaru_2009.tck'
    tracks, header = load_tck_file(subject, tck_file_name)
    save_to_mmvt(subject, tracks, header, 'Koganemaru')