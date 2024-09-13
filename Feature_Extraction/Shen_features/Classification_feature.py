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


def calculate_reho(file_path, cluster_size=27):
    """
    Calculate the ReHo for each voxel in the fMRI data.

    Parameters:
    - data: 4D numpy array of fMRI data (x, y, z, time)
    - cluster_size: Number of neighboring voxels to consider (default is 27, 3x3x3 cube)

    Returns:
    - reho_map: 3D numpy array of ReHo values
    """
    img = image.load_img(file_path)

    coords = [(1, 1, 1)]

    masker = input_data.NiftiSpheresMasker(seeds=coords, radius=3, detrend=True, standardize=True)
    timeseries = masker.fit_transform(img)

    return timeseries


'''
for i in range(1, x - 1):
    for j in range(1, y - 1):
        for k in range(1, z - 1):
            # Extract the time series for the current voxel and its neighbors
            cluster = data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2, :]
            cluster = cluster.reshape(-1, t)

            # Calculate Kendall's W (coefficient of concordance)
            concordance = 0
            for m in range(cluster.shape[0]):
                for n in range(m + 1, cluster.shape[0]):
                    concordance += kendalltau(cluster[m], cluster[n])[0]

            # Normalize by the number of pairs
            num_pairs = cluster_size * (cluster_size - 1) / 2
            reho_map[i, j, k] = concordance / num_pairs

return reho_map
'''

# Iterate over each voxel in the 3D space


type = calculate_reho(file_path, cluster_size=27)
print(type)
