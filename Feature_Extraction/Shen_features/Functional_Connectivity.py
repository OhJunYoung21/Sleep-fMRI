import urllib.request
import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import datasets
import os

# Download the Shen atlas
filename = '/Users/oj/Downloads/shen_2mm_268_parcellation.nii'

# Load the Shen atlas
atlas = datasets.fetch_atlas_yeo_2011(data_dir='/Users/oj/Downloads/atlas')

# Load the functional MRI data
data = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'
# Extract the signal from the functional MRI data using the Shen atlas
masker = NiftiMapsMasker(maps_img=atlas, standardize=True)
time_series = masker.fit_transform(data)

print(time_series.shape)

'''
# Compute the functional connectivity matrix
correlation_measure = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
correlation_matrix = correlation_measure.fit_transform([time_series][0])

# Visualize the functional connectivity matrix
print(correlation_matrix)
'''
