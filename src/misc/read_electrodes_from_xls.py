import os.path as op
from collections import defaultdict
from src.utils import utils
from src.utils import preproc_utils as pu
from src.examples import morph_electrodes_to_template

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def read_xls(xls_fname, subject_to='colin27'):
    template_system = 'mni'
    bipolar = True
    subjects_electrodes = defaultdict(list)
    electrodes_colors = defaultdict(list)
    for line in utils.xlsx_reader(xls_fname, skip_rows=1):
        subject, _, elec_name, _, anat_group = line
        subject = subject.replace('\'', '')
        if subject == '':
            break
        electrodes_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_morph_to_{}.txt'.format(subject_to))
        if op.isfile(electrodes_fname):
            elec_group, num1, num2 = utils.elec_group_number(elec_name, bipolar)
            if '{}{}-{}'.format(elec_group, num2, num1) != elec_name:
                num1, num2 = str(num1).zfill(2), str(num2).zfill(2)
            if '{}{}-{}'.format(elec_group, num2, num1) != elec_name:
                raise Exception('Wrong group or numbers!')
            for num in [num1, num2]:
                subjects_electrodes[subject].append('{}{}'.format(elec_group, num))
            electrodes_colors[subject].append((elec_name, int(anat_group)))
    subjects = list(subjects_electrodes.keys())
    electrodes = morph_electrodes_to_template.read_all_electrodes(subjects, False)
    template_electrodes = morph_electrodes_to_template.read_morphed_electrodes(
        electrodes, template_system, SUBJECTS_DIR, MMVT_DIR, True, subjects_electrodes, convert_to_bipolar=bipolar)
    morph_electrodes_to_template.save_template_electrodes_to_template(
        template_electrodes, bipolar, MMVT_DIR, template_system)
    morph_electrodes_to_template.export_into_csv(template_system, MMVT_DIR, bipolar)
    morph_electrodes_to_template.create_mmvt_coloring_file(template_system, template_electrodes, electrodes_colors)


if __name__ == '__main__':
    xls_fname = '/home/cashlab/Documents/noam/Onset_regions_for_illustration.xlsx'
    read_xls(xls_fname)