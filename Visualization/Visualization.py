import ast
import pandas as pd
import nilearn
from nilearn import plotting
from nilearn import input_data
import glob
import os
import numpy as np
from nilearn import image
import nibabel as nib
from nilearn.image import threshold_img
from Feature_Extraction.Shen_features.Classification_feature import FC_extraction, file_path, atlas_path
from Feature_Extraction.Shen_features.Classification_feature import region_alff_average
from Feature_Extraction.Shen_features.Classification_feature import FC_extraction
from Feature_Extraction.Shen_features.Classification_feature import atlas_path, file_path
from nilearn.image import new_img_like

reho_path_afni = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho/reho_01.nii.gz'
reho_path_CPAC = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-01/results/ReHo.nii.gz'
alff_path = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-01/results/alff.nii.gz'
falff_path = '/Users/oj/Desktop/Yoo_Lab/CPAC/HC/sub-01/results/falff.nii.gz'

reho_path_rbd = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-01/results/ReHo.nii.gz'
reho_path_hc = '/Users/oj/Desktop/Yoo_Lab/CPAC/HC/sub-01/results/ReHo.nii.gz'

alff_img = image.load_img(alff_path)
falff_img = image.load_img(falff_path)
reho_img_rbd = image.load_img(reho_path_rbd)
reho_img_hc = image.load_img(reho_path_hc)

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean',
                                          resampling_target="labels")

reho_imgs = glob.glob(os.path.join('/Users/oj/Desktop/CPAC_features', 'ReHo_*.nii.gz'))
reho_nifti = []

for k in reho_imgs:
    img = image.load_img(k)
    reho_nifti.append(img)

mean_img = image.mean_img(reho_nifti)
mean_data = shen_atlas.fit_transform(mean_img)

mean_masked = shen_atlas.inverse_transform(mean_data)
plotting.plot_stat_map(mean_masked)
plotting.show()


'''
reho_rbd_data = shen_atlas.fit_transform(reho_img_rbd)
reho_hc_data = shen_atlas.fit_transform(reho_img_hc)

rbd_img_masked = shen_atlas.inverse_transform(reho_rbd_data)
hc_img_masked = shen_atlas.inverse_transform(reho_hc_data)
'''
