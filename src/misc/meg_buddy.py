from MEEGbuddy import MEEGbuddy, create_demi_events
import glob
import re, os
import os.path as op
from pandas import read_csv, DataFrame
import numpy as np
from mne.io import Raw, RawArray
from mne import create_info, find_events

root = '/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/'


def get_data(subject, tasks=['MSIT'], modalities=['MEG'], exclude_subjects=['ep001']):
    out_files = np.load(op.join(root, 'file_list.npz'))['out_files'].item()
    data_struct = {task: {modality: {} for modality in ['EEG', 'MEG']} for task in out_files}
    for task in tasks:
        if task == 'Resting':
            stimuli = {'Demi': ['STI011', -1, 1]}
            response = None
            baseline = ['STI011', -2, -1]
        else:
            stimuli = {'Onset': ['STI001', -0.5, 1]}
            response = ['STI002', -1.5, 1]
            baseline = ['STI001', -1.1, -0.1]
        for modality in modalities:
            d_files = out_files[task][modality]
            # subjects = d_files.keys()
            # if not all([s in d_files for s in subjects]):
            #     raise ValueError('Behavior data mismatch subjects')
            # for subject in subjects:
            if subject not in d_files.keys():
                print('Subject not in d_files.keys!')
                continue
            if subject in exclude_subjects:
                print('Subject in exclude_subjects')
                continue
            if task == 'Resting':
                events_fname = os.path.join(os.getcwd(), 'Resting_Events', '%s_%s_events.npz' % (subject, modality))
                if os.path.isfile(events_fname):
                    behavior = np.load(events_fname)['behavior'].item()
                else:
                    print('Making demi events')
                    raw = Raw(d_files[subject], preload=True, verbose=False)
                    demi_events, demi_conditions = create_demi_events(raw, windows_length=1000, windows_shift=500)
                    behavior = {'Resting': [True for _ in range(demi_events.shape[0] - 6)]}
                    np.savez_compressed(events_fname, behavior=behavior)
                    info = create_info(['STI011'], raw.info['sfreq'], ['stim'], verbose=False)
                    arr = np.zeros((1, len(raw.times)))
                    for i in demi_events[5:-1, 0]:
                        arr[0, i - raw.first_samp:i + 100 - raw.first_samp] = 1
                    ch = RawArray(arr, info, verbose=False)
                    if 'STI011' in raw.ch_names:
                        raw.drop_channels(['STI011'])
                    raw.add_channels([ch], force_update_info=True)
                    events = find_events(raw, 'STI011')
                    if not all(demi_events[5:-1, 0] == events[:, 0]):
                        raise ValueError('Something went wrong with events')
                    raw.save(d_files[subject], overwrite=True)
                exclude = []
            else:
                behavior = {}
                df = read_csv(out_files[task]['behavior'][subject])
                if task == 'MSIT':
                    for category in list(df.columns):
                        behavior[category] = [df.loc[i, category] for i in range(len(df)) if
                                              df.loc[i, 'Stimuli'] != '+']
                    behavior['Condition'] = ['Congruent' if c == 1 else c for c in behavior['Condition']]
                    behavior['Condition'] = ['Incongruent' if c == 2 else c for c in behavior['Condition']]
                elif task == 'ECR':
                    for category in list(df.columns):
                        behavior[category] = [df.loc[i, category] for i in range(len(df))]
                if 'ResponseTime' in behavior:
                    exclude = [i for i in range(len(behavior['ResponseTime']))
                               if np.isnan(behavior['ResponseTime'][i])]
                else:
                    continue
            exclude_response = [0, len(behavior[list(behavior.keys())[0]]) - 1]
            print('Loading into MEEGbuddy %s %s %s' % (task, modality, subject))
            dir_path = '/autofs/space/karima_002/users/Resting' if task == 'Resting' else '/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam'
            data = MEEGbuddy(subject=subject, fdata=d_files[subject], behavior=behavior, eeg=modality == 'EEG',
                             meg=modality == 'MEG',
                             baseline=baseline, stimuli=stimuli, response=response, no_response=exclude,
                             task=task, exclude_response=exclude_response, tbuffer=0.3 if task == 'Resting' else 1,
                             subjects_dir=dir_path)
            data_struct[task][modality][subject] = data
    return data_struct