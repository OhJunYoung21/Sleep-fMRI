import ast
import pandas as pd
import nilearn
from nilearn import plotting
from nilearn import input_data
import glob
import os
import numpy as np
from nilearn import image
from nilearn import datasets
import nibabel as nib
from nilearn.image import threshold_img
from sklearn.metrics.pairwise import cosine_similarity

AAL = datasets.fetch_atlas_aal()
atlas_filename = AAL.maps

AAL_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean',
                                         resampling_target="labels")

alff_rbd_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/RBD/ReHo', 'ReHo_*.nii.gz'))
reho_rbd_nifti = []
alff_hc_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/HC/ReHo', 'ReHo_*.nii.gz'))
reho_hc_nifti = []

for k in alff_rbd_imgs:
    img = image.load_img(k)
    reho_rbd_nifti.append(img)

for k in alff_hc_imgs:
    img = image.load_img(k)
    reho_hc_nifti.append(img)

## 이미지의 평균을 계산하는 코드 ##

mean_rbd_img = image.mean_img(reho_rbd_nifti)
mean_rbd_data = AAL_atlas.fit_transform(mean_rbd_img)

mean_hc_img = image.mean_img(reho_hc_nifti)
mean_hc_data = AAL_atlas.fit_transform(mean_hc_img)

mean_rbd_masked = AAL_atlas.inverse_transform(mean_rbd_data)
mean_hc_masked = AAL_atlas.inverse_transform(mean_hc_data)


mean_rbd_vector = np.array(mean_rbd_masked.get_fdata().flatten())
mean_hc_vector = np.array(mean_hc_masked.get_fdata().flatten())

# 코사인 유사도 계산
similarity = cosine_similarity([mean_rbd_vector], [mean_hc_vector])

print(f"Cosine similarity: {similarity[0][0]}")





