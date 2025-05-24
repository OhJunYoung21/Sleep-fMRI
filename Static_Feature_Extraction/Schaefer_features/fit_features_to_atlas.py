import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
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
from nilearn import datasets


def fit_FC_atlas(path):
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=300)
    atlas_filename = schaefer.maps

    schaefer_masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    time_series = schaefer_masker.fit_transform(path)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])

    return correlation_matrix


### fit_feature_atlas : convert voxel unit data into region unit data

def fit_reho_atlas(reho_file):
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=300)

    atlas_filename = schaefer.maps  # .nii.gz 파일 경로

    schaefer_masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    reho_img = image.load_img(reho_file)

    masked_data = schaefer_masker.fit_transform([reho_img])

    return masked_data


def fit_alff_atlas(alff_path):
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=300)

    atlas_filename = schaefer.maps  # .nii.gz 파일 경로

    schaefer_masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    alff_img = image.load_img(alff_path)

    masked_data = schaefer_masker.fit_transform([alff_img])

    return masked_data


def fit_falff_atlas(falff_path):
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=300)

    atlas_filename = schaefer.maps  # .nii.gz 파일 경로

    schaefer_masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    falff_img = image.load_img(falff_path)

    masked_data = schaefer_masker.fit_transform([falff_img])

    return masked_data


def vector_to_symmetric_matrix(vec, n):
    mat = np.zeros((n, n))
    tril_indices = np.tril_indices(n, k=-1)
    mat[tril_indices] = vec
    mat = mat + mat.T  # 대칭으로 복원
    return mat


if __name__ == "__main__":
    fit_FC_atlas(path=None)
    fit_reho_atlas(reho_file=None)
    fit_alff_atlas(alff_path=None)
    fit_falff_atlas(falff_path=None)
