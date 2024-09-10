from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import os
from nilearn import image
from nilearn import plotting
from nilearn.signal import clean
from nilearn import input_data
import numpy as np
from nilearn import datasets
import nibabel as nib
from scipy.stats import rankdata
from nilearn.maskers import NiftiMasker
from scipy.ndimage import generate_binary_structure

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'


def FC_extraction(file_path):
    basc_atlas = datasets.fetch_atlas_basc_multiscale_2015()
    basc_atlas_name = basc_atlas["scale036"]

    data = image.load_img(file_path)

    masker = NiftiLabelsMasker(labels_img=basc_atlas_name, standardize=True)
    time_series = masker.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    return correlation_matrix


def ALFF_extraction(file_path, low_pass=0.08, high_pass=0.01, t_r=3.0):
    img = image.load_img(file_path)

    img_data = np.array(img.dataobj)

    cleaned_data = clean(img_data, low_pass=low_pass, high_pass=high_pass, detrend=True, standardize=True)

    alff_data = np.std(cleaned_data, axis=-1)

    alff_img = image.new_img_like(img, alff_data)

    return alff_img


def Reho_extraction(file_path):
    return


img = nib.load(file_path)

voxel_sizes = img.header.get_zooms()

print(voxel_sizes)
