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

    slice_orders = {
        35: [1, 19, 2, 20, 3, 21, 4, 22, 5, 23, 6, 24, 7, 25, 8, 26, 9, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 32, 15,
             33, 16, 34, 17, 35, 18],
        36: [1, 19, 2, 20, 3, 21, 4, 22, 5, 23, 6, 24, 7, 25, 8, 26, 9, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 32, 15,
             33, 16, 34, 17, 35, 18, 36],
        37: [1, 20, 2, 21, 3, 22, 4, 23, 5, 24, 6, 25, 7, 26, 8, 27, 9, 28, 10, 29, 11, 30, 12, 31, 13, 32, 14, 33, 15,
             34, 16, 35, 17, 36, 18, 37, 19],
        39: [1, 21, 2, 22, 3, 23, 4, 24, 5, 25, 6, 26, 7, 27, 8, 28, 9, 29, 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15,
             35, 16, 36, 17, 37, 18, 38, 19, 39, 20],
        40: [1, 21, 2, 22, 3, 23, 4, 24, 5, 25, 6, 26, 7, 27, 8, 28, 9, 29, 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15,
             35, 16, 36, 17, 37, 18, 38, 19, 39, 20, 40],
        41: [1, 22, 2, 23, 3, 24, 4, 25, 5, 26, 6, 27, 7, 28, 8, 29, 9, 30, 10, 31, 11, 32, 12, 33, 13, 34, 14, 35, 15,
             36, 16, 37, 17, 38, 18, 39, 19, 40, 20, 41, 21],
    }

    # Compute slice timings with three decimal places
    slice_timings = {N: [round((x - 1) * tr / N, 4) for x in order] for N, order in slice_orders.items()}

    return slice_timings[n_slices]

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
