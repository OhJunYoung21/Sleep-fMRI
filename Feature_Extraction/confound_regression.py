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

confounds = glob.glob(os.path.join(root_dir, 'sub-[0-9][0-9]', 'func',
                                   'sub-[0-9][0-9]_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv'))

temp_img = image.index_img(fMRI_img[0],10)
plotting.plot_img(temp_img)
plotting.show()
