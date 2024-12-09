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

dataset = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=325)
atlas_filename = dataset.maps


def FC_extraction(path):
    dynamic_FC = []

    basc_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    data = image.load_img(path)

    time_series = basc_atlas.fit_transform(data)

    window_size = 10
    step_size = 2
    window_number = ((time_series.shape)[0] - window_size) // step_size + 1

    for i in range(window_number):
        start = i * step_size
        end = start + window_size
        window_data = time_series[start:end]

        correlation_measure = ConnectivityMeasure(kind='correlation', standardize="zscore_sample")
        correlation_matrix = np.round(correlation_measure.fit_transform([window_data]), decimals=3)

        dynamic_FC.append(correlation_matrix)

    variance_matrix = np.round(np.var(dynamic_FC, axis=0), decimals=3)

    return variance_matrix


## calculate_3dReHo는 AFNI의 3dReHo를 사용해서 input으로는 4D image를 받고 output으로 3d image를 반환한다.


'''
건강군과 질병군마다 분류기준을 추출한다. 경로를 헷갈리지 않게 하기 위해서 feature 추출하는 함수를 2개씩 작성하였다.
'''


## region_reho_average는 mask가 나눈 region안의 voxel 값들의 평균을 계산한다.
def basc_reho_average(reho_path):
    ## 4D 데이터를 3D 데이터로 만들어준다.

    data = image.get_data(reho_path)

    variance = np.var(data, axis=3)

    alff_img = image.load_img(reho_path)

    variance_img = nib.Nifti2Image(variance, alff_img.affine)

    ## 3D 데이터에 BASC atlas를 씌워준다.

    BASC_atlas = input_data.NiftiLabelsMasker(labels_img=image.load_img(atlas_filename), standardize=True,
                                              strategy='mean',
                                              )

    masked_data = BASC_atlas.fit_transform([variance_img])

    return masked_data


def basc_alff_average(alff_path):
    ## 4D 데이터를 3D 데이터로 만들어준다.

    data = image.get_data(alff_path)

    variance = np.var(data, axis=3)

    alff_img = image.load_img(alff_path)

    variance_img = nib.Nifti2Image(variance, alff_img.affine)

    ## 3D데이터에 BASC atlas를 씌워준다.

    BASC_atlas = input_data.NiftiLabelsMasker(labels_img=image.load_img(atlas_filename), standardize=True,
                                              strategy='mean',
                                              )

    masked_data = BASC_atlas.fit_transform([variance_img])

    return masked_data


def basc_falff_average(falff_path):
    ## 4D 데이터를 3D 데이터로 만들어준다.

    data = image.get_data(falff_path)

    variance = np.var(data, axis=3)

    falff_img = image.load_img(falff_path)

    variance_img = nib.Nifti2Image(variance, falff_img.affine)

    ## 3D데이터에 BASC atlas를 씌워준다.

    BASC_atlas = input_data.NiftiLabelsMasker(labels_img=image.load_img(atlas_filename), standardize=True,
                                              strategy='mean',
                                              )

    masked_data = BASC_atlas.fit_transform([variance_img])

    return masked_data
