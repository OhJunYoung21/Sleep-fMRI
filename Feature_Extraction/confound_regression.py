import os
import glob
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.image import get_data
from nilearn.image import clean_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn import plotting

root_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/post_prep_RBD'

# 전처리가 끝난 fMRI 파일과 fMRIprep이 제공한 confound파일을 읽어온다.

fMRI_img = glob.glob(os.path.join(root_dir, 'sub-*', 'func',
                                  'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))

raw_confounds = glob.glob(os.path.join(root_dir, 'sub-[0-9][0-9]', 'func',
                                       'sub-[0-9][0-9]_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv'))

for index in range(len(fMRI_img)):
    # 혼란변수 파일 가져오기(pandas 사용)
    confounds = pd.read_csv(raw_confounds[index], sep='\t')
    # fMRI_img 가져오기
    fmri_img = fMRI_img[index]

    confounds_of_interest = confounds[
        ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
         'global_signal', 'white_matter', 'csf']
    ]

    cleaned_img = clean_img(fmri_img, confound=confounds_of_interest)


    cleaned_img.to_filename(f"{root_dir}/sub-{index}_confounds_regressed.nii.gz")
