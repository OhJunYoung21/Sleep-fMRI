import ast
import pandas as pd
import nilearn
from nilearn import plotting
from nilearn import input_data
import os
from nilearn import image
from Feature_Extraction.Shen_features.Classification_feature import FC_extraction, file_path, atlas_path

correlation_matrix = FC_extraction(file_path, atlas_path)

plotting.plot_matrix(correlation_matrix)
plotting.show()

reho_img = image.load_img('/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/reho/reho_01.nii.gz')
plotting.plot_img(reho_img)
plotting.show()
