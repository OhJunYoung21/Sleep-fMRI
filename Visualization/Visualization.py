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

AAL = datasets.fetch_atlas_aal()
atlas_filename = AAL.maps
labels = AAL.labels

AAL_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean',
                                         resampling_target="labels")

reho_rbd_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/RBD/ReHo', 'ReHo_*.nii.gz'))
reho_rbd = []
reho_hc_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/HC/ReHo', 'ReHo_*.nii.gz'))
reho_hc = []

for k in reho_rbd_imgs:
    img = image.load_img(k)
    reho_rbd.append(img)

for k in reho_hc_imgs:
    img = image.load_img(k)
    reho_hc.append(img)

mean_rbd_img = image.mean_img(reho_rbd)
mean_rbd_data = AAL_atlas.fit_transform(mean_rbd_img)
mean_hc_img = image.mean_img(reho_hc)
mean_hc_data = AAL_atlas.fit_transform(mean_hc_img)

mean_rbd_masked = AAL_atlas.inverse_transform(mean_rbd_data)
mean_hc_masked = AAL_atlas.inverse_transform(mean_hc_data)
plotting.plot_stat_map(mean_rbd_masked, title="RBD_mean")
plotting.show()
plotting.plot_stat_map(mean_hc_masked, title="HC_mean")
plotting.show()
