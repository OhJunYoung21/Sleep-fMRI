import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import datasets
import os
from nilearn import image
from nilearn import input_data
from scipy.stats import kendalltau
from nipype.interfaces import afni

# Download the Shen atlas
atlas_path = '/Users/oj/Downloads/shen_2mm_268_parcellation.nii'

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'


def FC_extraction(file_path, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    data = image.load_img(file_path)

    time_series = shen_atlas.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return correlation_matrix


def calculate_3dReHo(file_path):
    reho = afni.ReHo()

    reho.inputs.in_file = file_path
    reho.inputs.out_file = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho_1.nii.gz'

    result = reho.run()

    result_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho_1.nii.gz'

    img = image.load_img(result_path)

    return img


def region_reho(reho_file, atlas_path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    reho_img = image.load_img(reho_file)

    masked_data = shen_atlas.fit_transform([reho_img])

    return masked_data


reho_file = calculate_3dReHo(file_path)

result = region_reho(reho_file, atlas_path)

print(result.shape)
