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
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import levene

Schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200)
atlas_filename = Schaefer.maps
labels = Schaefer.labels

Schaefer_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean',
                                              resampling_target="labels")

reho_rbd_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/RBD/ReHo', 'ReHo_*.nii.gz'))
reho_rbd = []
reho_hc_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/HC/ReHo', 'ReHo_*.nii.gz'))
reho_hc = []

for k in reho_rbd_imgs:
    img = image.load_img(k)
    rbd = Schaefer_atlas.fit_transform(img)
    reho_rbd.append(rbd)

for k in reho_hc_imgs:
    img = image.load_img(k)
    hc = Schaefer_atlas.fit_transform(img)
    reho_hc.append(hc)

reho_rbd = np.array([item[0] for item in reho_rbd])
reho_hc = np.array([item[0] for item in reho_hc])


# 정규분포를 따르는 region과 그렇지 않은 region의 노드를 알려준다.

def check_normality(features):
    mann_whitneyu = []
    t_test = []

    for i in range(200):
        stat, p_val = shapiro(features[:, i])

        if p_val < 0.05:
            mann_whitneyu.append(i)
        else:
            t_test.append(i)

    return mann_whitneyu, t_test


'''
alff_data = pd.read_excel('alff_final_data.xlsx')

no_follow_normality = alff_data['no_follow_normality'].tolist()
significant_diff = alff_data['significant_diff'].tolist()

no_follow_index = next((i for i, x in enumerate(no_follow_normality) if np.isnan(x)), len(no_follow_normality))
no_follow_normality = no_follow_normality[:no_follow_index]

significant_diff_mann = []

alff_rbd = np.array([item[0] for item in alff_rbd])
alff_hc = np.array([item[0] for item in alff_hc])

no_follow_normality = [int(i) for i in no_follow_normality]

for i in no_follow_normality:
    u_stats, p_value = mannwhitneyu(alff_rbd[:, i], alff_hc[:, i])  # i번째 열에 대해 Shapiro-Wilk 검정 수행

    if p_value > 0.05:
        print(f"Feature {i}: 유의미한 차이를 보이지 않는다. (p-value={p_value:.3f})")

    else:
        print(f"Feature {i}: 유의미한 차이를 보인다. (p-value={p_value:.3f})")
        significant_diff_mann.append(i)

alff_data['significant_diff_mann'] = pd.Series(significant_diff_mann)

alff_data.to_excel('alff_diff_data.xlsx', index=False)
'''
