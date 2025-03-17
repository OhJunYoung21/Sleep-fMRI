import numpy as np
import json
import glob as glob
import os

slice_time = 3 / 35

json_slice_36 = (np.array(
    [1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29,
     35, 6, 12, 18, 24, 30, 36]) - 1) * slice_time

json_slice_35 = (np.array(
    [1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29,
     35, 6, 12, 18, 24, 30, ]) - 1) * slice_time

json_slice_37 = (np.array(
    [1, 7, 13, 19, 25, 31, 37, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29,
     35, 6, 12, 18, 24, 30, 36]) - 1) * slice_time

json_slice_41 = (np.array(
    [1, 7, 13, 19, 25, 31, 37, 2, 8, 14, 20, 26, 32, 38, 3, 9, 15, 21, 27, 33, 39, 4, 10, 16, 22, 28, 34, 40, 5, 11, 17,
     23,
     29,
     35, 41, 6, 12, 18, 24, 30, 36]) - 1) * slice_time

print(np.round(json_slice_36, 4).tolist())
print(np.round(json_slice_37, 4).tolist())
print(np.round(json_slice_41, 4).tolist())
print(np.round(json_slice_35, 4).tolist())

'''

def add_info_to_json(file_path, key, value):
    """
    Adds or updates a specific key-value pair in a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.
    - key (str): The key to add/update.
    - value (str): The value to associate with the key.
    """
    try:
        # Try to load existing data
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, initialize as an empty dict
        data = {}

    # Add or update the key-value pair
    data[key] = value

    # Save the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"✅ Successfully added '{key}: {value}' to {file_path}")


root_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS'

json_path = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
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

for json_file in json_path:
    add_info_to_json(json_file, "SliceTiming", str(json_slice))


'''
