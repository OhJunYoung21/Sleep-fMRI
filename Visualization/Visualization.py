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
from scipy.stats import ttest_ind
from Feature_Extraction.BASC_features.BASC_features import FC_extraction

BASC_atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=325)
atlas_filename = BASC_atlas.maps

AAL_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean',
                                         resampling_target="labels")
'''
rbd_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/RBD/fALFF', 'falff_*.nii.gz'))
rbd = []
hc_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/HC/fALFF', 'falff_*.nii.gz'))
hc = []

for k in rbd_imgs:
    img = image.load_img(k)
    rbd.append(img)

for k in hc_imgs:
    img = image.load_img(k)
    hc.append(img)
'''

matrix_RBD = FC_extraction(
    '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz')
plotting.plot_matrix(matrix_RBD, title='RBD_connectivity')
matrix_HC = FC_extraction(
    '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/sub-01_confounds_regressed.nii.gz')
plotting.plot_matrix(matrix_HC, title='HC_connectivity')
plotting.show()

'''
mean_rbd_img = image.mean_img(rbd)
mean_rbd_data = AAL_atlas.fit_transform(mean_rbd_img)
mean_hc_img = image.mean_img(hc)
mean_hc_data = AAL_atlas.fit_transform(mean_hc_img)

mean_rbd_masked = AAL_atlas.inverse_transform(mean_rbd_data)
mean_hc_masked = AAL_atlas.inverse_transform(mean_hc_data)
plotting.plot_stat_map(mean_rbd_masked, title="RBD_falff_mean", vmax=0.9)
plotting.show()
plotting.plot_stat_map(mean_hc_masked, title="HC_falff_mean", vmax=0.9)
plotting.show()
'''
