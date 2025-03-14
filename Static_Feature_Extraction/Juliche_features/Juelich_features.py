import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import datasets
import os
import nibabel as nib
from nilearn import masking
from nilearn import image
from nilearn import input_data
from nipype.interfaces import afni
from scipy import stats

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'

dataset = datasets.fetch_atlas_juelich('maxprob-thr0-2mm')
atlas_filename = dataset.maps
labels = dataset.labels


def FC_extraction(path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    data = image.load_img(path)

    time_series = shen_atlas.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return correlation_matrix


## calculate_3dReHo는 AFNI의 3dReHo를 사용해서 input으로는 4D image를 받고 output으로 3d image를 반환한다.


'''
건강군과 질병군마다 분류기준을 추출한다. 경로를 헷갈리지 않게 하기 위해서 feature 추출하는 함수를 2개씩 작성하였다.
'''


## region_reho_average는 mask가 나눈 region안의 voxel 값들의 평균을 계산한다.
def juelich_reho_average(reho_file):
    juelich_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean')

    reho_img = image.load_img(reho_file)

    masked_data = juelich_atlas.fit_transform([reho_img])

    return masked_data


def juelich_alff_average(alff_path):
    juelich_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, strategy='mean',
                                                 resampling_target="labels")

    alff_img = image.load_img(alff_path)

    masked_data = juelich_atlas.fit_transform([alff_img])

    return masked_data
