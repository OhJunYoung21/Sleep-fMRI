import os
import nilearn
from nilearn import image
import glob
import re
import numpy as np
import json

root_dir = "/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_post_BIDS"


def generate_slice_timing(n_slices, tr):
    """
    Generate SliceTiming array based on interleaved descending slice acquisition order.

    Parameters:
        n_slices (int): Number of slices.
        tr (float): Repetition Time (in seconds).

    Returns:
        list: SliceTiming values in seconds, ordered by slice number (1-based).
    """

    # Generate interleaved descending order
    # Odd positions descending: n, n-2, ...
    # Even positions descending: n-1, n-3, ...

    if n_slices % 2 == 0:
        even = list(range(n_slices, 0, -2))
        odd = list(range(n_slices - 1, 0, -2))
        slice_order = even + odd
    else:
        odd = list(range(n_slices, 0, -2))
        even = list(range(n_slices - 1, 0, -2))
        slice_order = odd + even

    # Create slice timing list in slice-number order (index = slice_num - 1)
    slice_timing = [0] * n_slices
    for i, slice_num in enumerate(slice_order):
        slice_timing[slice_num - 1] = round((i - 1) * (3 / n_slices), 4)  # round for JSON compatibility

    return slice_timing


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
        add_info_to_json(json_file, "SliceTiming", generate_slice_timing(35, 3))
    elif test_img.shape[2] == 36:
        add_info_to_json(json_file, "SliceTiming", generate_slice_timing(36, 3))
    elif test_img.shape[2] == 37:
        add_info_to_json(json_file, "SliceTiming", generate_slice_timing(37, 3))
    elif test_img.shape[2] == 39:
        add_info_to_json(json_file, "SliceTiming", generate_slice_timing(39, 3))
    elif test_img.shape[2] == 40:
        add_info_to_json(json_file, "SliceTiming", generate_slice_timing(40, 3))
    elif test_img.shape[2] == 41:
        add_info_to_json(json_file, "SliceTiming", generate_slice_timing(41, 3))
    else:
        continue
