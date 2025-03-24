import os
import nilearn
from nilearn import image
import glob
import re
import numpy as np
import json

root_dir = "/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS_test"

slice_time = 3 / 35

json_slice_36 = (np.array(
    [1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29,
     35, 6, 12, 18, 24, 30, 36]) - 1) * slice_time

json_slice_35 = (np.array(
    [1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29,
     35, 6, 12, 18, 24, 30]) - 1) * slice_time

json_slice_37 = (np.array(
    [1, 7, 13, 19, 25, 31, 37, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29,
     35, 6, 12, 18, 24, 30, 36]) - 1) * slice_time

json_slice_41 = (np.array(
    [1, 7, 13, 19, 25, 31, 37, 2, 8, 14, 20, 26, 32, 38, 3, 9, 15, 21, 27, 33, 39, 4, 10, 16, 22, 28, 34, 40, 5, 11, 17,
     23,
     29,
     35, 41, 6, 12, 18, 24, 30, 36]) - 1) * slice_time

test_file_nifti = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                         'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_rec-RESEARCHMRI_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_bold.nii.gz'))

test_file_json = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                        'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_bold.json')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_bold.json')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_bold.json')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_bold.json')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_bold.json')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_rec-RESEARCHMRI_bold.json')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_bold.json'))

test_file_json = sorted(test_file_json)

subject_slice_number = {}


# subject가 몇개의 슬라이스를 가지고 있는지를 알려준다.

def add_info_to_json(file_path, key, value):
    """
    Adds or updates a specific key-value pair in a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.
    - key (str): The key to add/update.
    - value (str): The value to associate with the key.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"✅ Successfully added '{key}: {value}' to {file_path}")


for number, json_file in zip(sorted(test_file_nifti), sorted(test_file_json)):
    test_img = image.load_img(number)

    if test_img.shape[2] == 35:
        add_info_to_json(json_file, "SliceTiming", np.round(json_slice_35, 4).tolist())
    elif test_img.shape[2] == 36:
        add_info_to_json(json_file, "SliceTiming", np.round(json_slice_36, 4).tolist())
    elif test_img.shape[2] == 37:
        add_info_to_json(json_file, "SliceTiming", np.round(json_slice_37, 4).tolist())
    elif test_img.shape[2] == 41:
        add_info_to_json(json_file, "SliceTiming", np.round(json_slice_41, 4).tolist())

    '''
    match = re.search(r'sub-(\d+)', number)
    subject_slice_number[match.group(1)] = image.load_img(number).shape[2]
    '''
