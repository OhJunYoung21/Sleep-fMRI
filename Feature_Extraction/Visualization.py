import ast
import pandas as pd
import nilearn
from nilearn import plotting
from nilearn import input_data
import os
from nilearn import image
from nilearn.image import threshold_img
from Feature_Extraction.Shen_features.Classification_feature import FC_extraction, file_path, atlas_path

file2_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-02_confounds_regressed.nii.gz'

alff_path = '/Users/oj/Desktop/Yoo_Lab/CPAC/HC/sub-09/results/alff.nii.gz'

alff_img = image.load_img(alff_path)

threshold_percentile_img = threshold_img(alff_img, threshold="99%", copy=False)

plotting.plot_stat_map(
    threshold_percentile_img,
    display_mode="z",
    cut_coords=5,
    title="Threshold image with string percentile",
    colorbar=False,
)

plotting.show()
