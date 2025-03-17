import os
import nilearn
from nilearn import image
import glob
import re

root_dir = "/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS_test"

test_file = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
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

for j in sorted(test_file):
    test_img = image.load_img(j)

    match = re.search(r'sub-(\d+)', j)
    print(match.group(1), image.load_img(j).shape[2])
