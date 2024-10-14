import ast
import pandas as pd
import nilearn
from nilearn import plotting
from nilearn import input_data
import os
from nilearn import image
import nibabel as nib
from nilearn.image import threshold_img
from Feature_Extraction.Shen_features.Classification_feature import FC_extraction, file_path, atlas_path
from Feature_Extraction.Shen_features.Classification_feature import region_alff_average


alff_path = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-09/results/alff.nii.gz'

alff_img = image.load_img(alff_path)

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean',
                                          resampling_target="labels")

data = shen_atlas.fit_transform(alff_img)


alff_img_masked = shen_atlas.inverse_transform(data)

plotting.plot_stat_map(alff_img_masked, title="Shen_ALFF")
plotting.show()



