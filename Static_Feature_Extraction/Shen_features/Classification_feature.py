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
from nilearn.image import math_img
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img

# Download the Shen atlas
atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

shen = image.load_img(atlas_path)


## 특정 지역과 다른 모든 지역간의 상관계수를 계산하여 더한다. 이는 해당 특정 지역이 다른 지역들과 얼마나 유사한 변화양상(BOLD signal)을 띄는지 측정할 수 있다.


def FC_extraction(path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    data = image.load_img(path)

    time_series = shen_atlas.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])

    return correlation_matrix


## region_reho_average는 mask가 나눈 region안의 voxel 값들의 평균을 계산한다.
def region_reho_average(reho_file, atlas):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas, standardize=True, strategy='mean')

    reho_img = image.load_img(reho_file)

    masked_data = shen_atlas.fit_transform([reho_img])

    return masked_data


def region_alff_average(alff_path, atlas):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas, standardize=True, strategy='mean',
                                              resampling_target="labels")

    alff_img = image.load_img(alff_path)

    masked_data = shen_atlas.fit_transform([alff_img])

    return masked_data


if __name__ == "__main__":
    FC_extraction(path=None)
    region_reho_average(reho_file=None, atlas=None)
    region_alff_average(alff_path=None, atlas=None)
