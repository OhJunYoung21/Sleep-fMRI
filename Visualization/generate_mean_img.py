import numpy as np
import glob
from nilearn.image import mean_img
from nilearn import image, input_data
from nilearn import plotting
from Feature_Extraction.Shen_features.Classification_feature import atlas_path
from scipy import stats

reho_sub_1 = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-01/results/ReHo.nii.gz'
reho_sub_2 = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-02/results/ReHo.nii.gz'

result_1 = image.load_img(reho_sub_1)
result_2 = image.load_img(reho_sub_2)

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean',
                                          resampling_target="labels")

reho_rbd_1 = shen_atlas.fit_transform(result_1)
reho_rbd_2 = shen_atlas.fit_transform(result_2)

rbd_img_1 = shen_atlas.inverse_transform(reho_rbd_1)
rbd_img_2 = shen_atlas.inverse_transform(reho_rbd_2)

t_stat, p_value = stats.ttest_ind(rbd_img_1, rbd_img_2)

print(f"T-statistic: {t_stat}")
print(f"p-value: {p_value}")
