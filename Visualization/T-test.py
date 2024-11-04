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

alff_rbd_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/RBD/ReHo', 'ReHo_*.nii.gz'))
reho_rbd = []
alff_hc_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/HC/ReHo', 'ReHo_*.nii.gz'))
reho_hc = []

for k in alff_rbd_imgs:
    img = image.load_img(k)
    aal_rbd = AAL_atlas.fit_transform(img)
    reho_rbd.append(aal_rbd)

for k in alff_hc_imgs:
    img = image.load_img(k)
    aal_hc = AAL_atlas.fit_transform(img)
    reho_hc.append(aal_hc)

t_statistics = []
p_values = []

reho_rbd = np.array([item[0] for item in reho_rbd])
reho_hc = np.array([item[0] for item in reho_hc])

for i in range(116):
    t_stat, p_val = ttest_ind(reho_rbd[:, i], reho_hc[:, i])
    t_statistics.append(t_stat)
    p_values.append(p_val)

# 결과 출력
for i in range(116):
    print(f"Feature {i + 1}: T-statistic = {t_statistics[i]:.3f}, p-value = {p_values[i]:.3f}") if p_values[
                                                                                                       i] <= 0.05 else print(
        "None significant")
