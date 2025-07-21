import os
import glob
import numpy as np
import pandas as pd
from nilearn.image import clean_img
import re

root_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_prep_res_2'

# 전처리가 끝난 fMRI 파일과 fMRIprep이 제공한 confound파일을 읽어온다.


fMRI_img = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                  'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152Lin_res-2_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152Lin_res-2_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152Lin_res-2_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_space-MNI152Lin_res-2_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_space-MNI152Lin_res-2_desc-preproc_bold.nii.gz')) + glob.glob(
    os.path.join(root_dir, 'sub-*', 'func',
                 'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_space-MNI152Lin_res-2_desc-preproc_bold.nii.gz'))

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

sample = pd.read_csv(raw_confounds[0], sep='\t')

for index in range(len(fMRI_img)):
    # fMRI_image와 confounds 업로드

    fmri_img = fMRI_img[index]
    confounds = pd.read_csv(raw_confounds[index], sep='\t')

    confounds = confounds.fillna(0)  # NaN 값 채우기
    confounds = confounds.replace([np.inf, -np.inf], 0)

    # 정규표현식을 사용해서 subject_number를 얻어온다. index+1로 얻어오는 경우, subject가 7,9인 경우 9를 8로 적어버리는 오류가 발생한다.
    subject_number = re.search(r'sub-(\d+)', fMRI_img[index]).group(1)

    cleaned_image = clean_img(fmri_img, confounds=confounds[[
        'global_signal', 'csf', 'white_matter',
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z',
        'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
        'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
        'global_signal_derivative1', 'csf_derivative1', 'white_matter_derivative1'
    ]],
                              detrend=True, standardize=True)

    cleaned_image.to_filename(
        f"/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_confound_regressed_res_2_18_parameters/sub-{subject_number}_confounds_regressed.nii.gz")
