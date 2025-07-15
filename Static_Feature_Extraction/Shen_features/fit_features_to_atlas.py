import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import image
from nilearn import input_data
from nipype.interfaces import afni
from scipy import stats
from nilearn.image import math_img
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img

# Download the Shen atlas
atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen = image.load_img(atlas_path)


## 특정 지역과 다른 모든 지역간의 상관계수를 계산하여 더한다. 이는 해당 특정 지역이 다른 지역들과 얼마나 유사한 변화양상(BOLD signal)을 띄는지 측정할 수 있다.


def FC_for_shen(path):
    shen_masker = input_data.NiftiLabelsMasker(labels_img=shen, standardize=True, strategy='mean', low_pass=0.1,
                                               high_pass=0.01, t_r=3,
                                               resampling_target="labels")
    data = image.load_img(path)

    time_series = shen_masker.fit_transform(data)

    print(time_series.shape)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])

    return correlation_matrix


### fit_feature_atlas : convert voxel unit data into region unit data

def reho_for_shen(reho_path):
    shen_masker = input_data.NiftiLabelsMasker(labels_img=shen, standardize=True, strategy='mean',
                                               resampling_target="labels")

    reho_img = image.load_img(reho_path)

    masked_data = shen_masker.fit_transform([reho_img])

    return masked_data


def alff_for_shen(alff_path):
    shen_masker = input_data.NiftiLabelsMasker(labels_img=shen, standardize=True, strategy='mean',
                                               resampling_target="labels")

    alff_img = image.load_img(alff_path)

    masked_data = shen_masker.fit_transform([alff_img])

    return masked_data


def falff_for_shen(falff_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=shen, standardize=True, strategy='mean',
                                              resampling_target="labels")

    falff_img = image.load_img(falff_path)

    masked_data = shen_atlas.fit_transform([falff_img])

    return masked_data


def vector_to_symmetric_matrix(vec, n):
    mat = np.zeros((n, n))
    tril_indices = np.tril_indices(n, k=-1)
    mat[tril_indices] = vec
    mat = mat + mat.T  # 대칭으로 복원
    return mat


if __name__ == "__main__":
    FC_for_shen(path=None)
    reho_for_shen(reho_path=None)
    alff_for_shen(alff_path=None)
    falff_for_shen(falff_path=None)
