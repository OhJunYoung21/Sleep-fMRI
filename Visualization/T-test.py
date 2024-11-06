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


def check_variance():
    mann_whitneyu, t_test = check_normality(reho_rbd)

    welch = []
    student = []

    for j in t_test:
        t_stats, p_val = levene(reho_rbd[:, j], reho_hc[:, j])

        # 두그룹의 분산이 동일하지 않은 경우.
        if p_val < 0.05:
            welch.append(j)
        else:
            student.append(j)
    return welch, student


mann_whitneyu_check, t_test_check = check_normality(reho_rbd)

welch, student = check_variance()


def welch_t_test(welch_list):
    result_welch = []

    for j in welch_list:
        t_stats, p_val = ttest_ind(reho_rbd[:, j], reho_hc[:, j], equal_var=False)

        if p_val < 0.05:
            result_welch.append(j)
        else:
            continue
    return result_welch


def student_t_test(student_list):
    result_student = []

    for j in student_list:
        t_stats, p_val = ttest_ind(reho_rbd[:, j], reho_hc[:, j], equal_var=True)

        if p_val < 0.05:
            result_student.append(j)
        else:
            continue
    return result_student


def mann_whitney_test(mann_list):
    result_mann = []

    for j in mann_list:
        u_stats, p_val = mannwhitneyu(reho_rbd[:, j], reho_hc[:, j], alternative='two-sided')

        if p_val < 0.05:
            result_mann.append(j)
        else:
            continue
    return result_mann


student_test = student_t_test(student)
welch_test = welch_t_test(welch)
mann_test = mann_whitney_test(mann_whitneyu_check)
reho_Schaefer_data = pd.DataFrame(
    {
        'follow_normality': pd.Series(t_test_check),
        'no_follow_normality': pd.Series(mann_whitneyu_check),
        'welch': pd.Series(welch_test),
        'student': pd.Series(student_test),
        'mann_whitneyu': pd.Series(mann_test),
    }
)

reho_Schaefer_data.to_excel('reho_Schaefer_data.xlsx')

# 두그룹의 분산이 동일한 경우
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
