from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import os
from nilearn import image
from nilearn import plotting

basc_atlas = datasets.fetch_atlas_basc_multiscale_2015()
basc_atlas_name = basc_atlas["scale036"]

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'
data = image.load_img(file_path)

masker = NiftiLabelsMasker(labels_img=basc_atlas_name, standardize=True)
time_series = masker.fit_transform(data)

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

plotting.plot_matrix(correlation_matrix)
plotting.show()