import os
import glob
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.image import get_data
from nilearn.image import clean_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn import plotting
import re

root_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_positive'

# 전처리가 끝난 fMRI 파일과 fMRIprep이 제공한 confound파일을 읽어온다.


fMRI_img = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                  'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))

raw_confounds = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                       'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_desc-confounds_timeseries.tsv')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_desc-confounds_timeseries.tsv'))

fMRI_img = sorted(fMRI_img)

raw_confounds = sorted(raw_confounds)

for index in range(len(fMRI_img)):
    # fMRI_image와 confounds 업로드

    fmri_img = fMRI_img[index]
    confounds = pd.read_csv(raw_confounds[index], sep='\t')

    confounds_of_interest = confounds[
        ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
         'global_signal', 'white_matter', 'csf']
    ]
    # 정규표현식을 사용해서 subject_number를 얻어온다. index+1로 얻어오는 경우, subject가 7,9인 경우 9를 8로 적어버리는 오류가 발생한다.
    subject_number = re.search(r'sub-(\d+)', fMRI_img[index]).group(1)

    cleaned_image = clean_img(fmri_img, confounds=confounds_of_interest)

    cleaned_image.to_filename(
        f"/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/sub-{subject_number}_confounds_regressed.nii.gz")
