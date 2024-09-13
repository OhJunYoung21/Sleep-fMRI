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


def Reho_extraction(file_path):
    func = image.load_img(file_path)

    return


def calculate_reho(data, cluster_size=27):
    """
    Calculate the ReHo for each voxel in the fMRI data.

    Parameters:
    - data: 4D numpy array of fMRI data (x, y, z, time)
    - cluster_size: Number of neighboring voxels to consider (default is 27, 3x3x3 cube)

    Returns:
    - reho_map: 3D numpy array of ReHo values
    """
    x, y, z, t = data.shape
    reho_map = np.zeros((x, y, z))

    example_data = data[1, 1, 1, :]

    # Iterate over each voxel in the 3D space
    return example_data


data = image.load_img(file_path)

reho_map = calculate_reho(data, cluster_size=27)
print(reho_map)
