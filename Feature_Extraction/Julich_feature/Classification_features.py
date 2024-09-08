import nilearn
from nilearn import image
import numpy as np
import pandas as pd
import os

## confound_regressed된 파일 업로드

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'

rbd_image = image.load_img(file_path)

print(rbd_image.affine)
