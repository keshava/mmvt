import os.path as op
import nibabel as nib
import numpy as np
from collections import defaultdict
from src.utils import utils
from src.utils import preproc_utils as pu
from src.examples import morph_electrodes_to_template
from src.examples import ela_morph_electrodes

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def read_xls(xls_fname, subject_to='colin27', atlas='aparc.DKTatlas', check_morph_file=False):
    bipolar = True
    template_header = nib.load(op.join(SUBJECTS_DIR, subject_to, 'mri', 'T1.mgz')).header
    subjects_electrodes = defaultdict(list)
    electrodes_colors = defaultdict(list)
    for line in utils.xlsx_reader(xls_fname, skip_rows=1):
        subject, _, elec_name, _, anat_group = line
        subject = subject.replace('\'', '')
        if subject == '':
            break
        if check_morph_file:
            electrodes_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_morph_to_{}.txt'.format(subject_to))
            if not op.isfile(electrodes_fname):
                continue
        elec_group, num1, num2 = utils.elec_group_number(elec_name, bipolar)
        if '{}{}-{}'.format(elec_group, num2, num1) != elec_name:
            num1, num2 = str(num1).zfill(2), str(num2).zfill(2)
        if '{}{}-{}'.format(elec_group, num2, num1) != elec_name:
            raise Exception('Wrong group or numbers!')
        for num in [num1, num2]:
            subjects_electrodes[subject].append('{}{}'.format(elec_group, num))
        electrodes_colors[subject].append((elec_name, int(anat_group)))
    subjects = list(subjects_electrodes.keys())
    for subject in subjects:
        if utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{hemi}.aparc.DKTatlas.annot')):
            atlas = 'aparc.DKTatlas'
        elif utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{hemi}.aparc.DKTatlas40.annot')):
            atlas = 'aparc.DKTatlas40'
        else:
            print('No atlas for {}!'.format(atlas))
            continue
        try:
            ela_morph_electrodes.calc_elas(
                subject,  subjects_electrodes[subject], subject_to, template_header, bipolar=False, atlas=atlas)
        except:
            utils.print_last_error_line()
            continue


def read_morphed_electrodes(xls_fname, subject_to='colin27', bipolar=True):
    subjects_electrodes = defaultdict(list)
    for line in utils.xlsx_reader(xls_fname, skip_rows=1):
        subject, _, elec_name, _, anat_group = line
        subject = subject.replace('\'', '')
        if subject == '':
            break
        elec_group, num1, num2 = utils.elec_group_number(elec_name, bipolar)
        if '{}{}-{}'.format(elec_group, num2, num1) != elec_name:
            num1, num2 = str(num1).zfill(2), str(num2).zfill(2)
        if '{}{}-{}'.format(elec_group, num2, num1) != elec_name:
            raise Exception('Wrong group or numbers!')
        elecs_pos = []
        for num in num1, num2:
            elec_input_fname = op.join(MMVT_DIR, subject, 'electrodes', '{}{}_ela_morphed.npz'.format(elec_group, num))
            d = np.load(elec_input_fname)
            elecs_pos.append(d['pos'])
        bipolar_ele_pos = np.mean(elecs_pos, axis=0)
        subjects_electrodes[subject].append()
        print('sdf')

def morph_csv():
    electrodes = morph_electrodes_to_template.read_all_electrodes(subjects, False)
    template_electrodes = morph_electrodes_to_template.read_morphed_electrodes(
        electrodes, template_system, SUBJECTS_DIR, MMVT_DIR, True, subjects_electrodes, convert_to_bipolar=bipolar)
    morph_electrodes_to_template.save_template_electrodes_to_template(
        template_electrodes, bipolar, MMVT_DIR, template_system)
    morph_electrodes_to_template.export_into_csv(template_system, MMVT_DIR, bipolar)
    morph_electrodes_to_template.create_mmvt_coloring_file(template_system, template_electrodes, electrodes_colors)


if __name__ == '__main__':
    fols = ['C:\\Users\\peled\\Documents\\Pariya', '/home/cashlab/Documents/noam/', '/home/npeled/Documents/pyraya']
    fol = [f for f in fols if op.isdir(f)][0]
    xls_fname = op.join(fol, 'Onset_regions_for_illustration.xlsx')

    # read_xls(xls_fname)
    read_morphed_electrodes(xls_fname, subject_to='colin27')