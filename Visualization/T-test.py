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

falff_rbd_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/RBD/fALFF', 'falff_*.nii.gz'))
falff_rbd = []
falff_hc_imgs = glob.glob(os.path.join('/Users/oj/Desktop/Yoo_Lab/CPAC_features/HC/fALFF', 'falff_*.nii.gz'))
falff_hc = []

for k in falff_rbd_imgs:
    img = image.load_img(k)
    rbd = Schaefer_atlas.fit_transform(img)
    falff_rbd.append(rbd)

for k in falff_hc_imgs:
    img = image.load_img(k)
    hc = Schaefer_atlas.fit_transform(img)
    falff_hc.append(hc)

falff_rbd = np.array([item[0] for item in falff_rbd])
falff_hc = np.array([item[0] for item in falff_hc])


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


def check_variance():
    mann_whitneyu, t_test = check_normality(falff_rbd)

    welch = []
    student = []

    for j in t_test:
        t_stats, p_val = levene(falff_rbd[:, j], falff_hc[:, j])

        # 두그룹의 분산이 동일하지 않은 경우.
        if p_val < 0.05:
            welch.append(j)
        else:
            student.append(j)
    return mann_whitneyu, welch, student


mann_whitneyu, welch, student = check_variance()


def welch_t_test(welch_list):
    result_welch = []

    for j in welch_list:
        t_stats, p_val = ttest_ind(falff_rbd[:, j], falff_hc[:, j], equal_var=False)

        if p_val < 0.05:
            result_welch.append(j)
        else:
            continue
    return result_welch


def student_t_test(student_list):
    result_student = []

    for j in student_list:
        t_stats, p_val = ttest_ind(falff_rbd[:, j], falff_hc[:, j], equal_var=True)

        if p_val < 0.05:
            result_student.append(j)
        else:
            continue
    return result_student


def mann_whitney_test(mann_list):
    result_mann = []

    for j in mann_list:
        u_stats, p_val = mannwhitneyu(falff_rbd[:, j], falff_hc[:, j], alternative='two-sided')

        if p_val < 0.05:
            result_mann.append(j)
        else:
            continue
    return result_mann


student_test = student_t_test(student)
welch_test = welch_t_test(welch)
mann_test = mann_whitney_test(mann_whitneyu)
reho_Schaefer_data = pd.DataFrame(
    {
        'welch': pd.Series(welch_test),
        'student': pd.Series(student_test),
        'mann_whitneyu': pd.Series(mann_test),
    }
)

reho_Schaefer_data.to_excel('Schaefer_data.xlsx')
