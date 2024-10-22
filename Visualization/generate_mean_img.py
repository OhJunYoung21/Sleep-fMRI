import numpy as np
import glob
from nilearn.image import mean_img
from nilearn import image, input_data
from nilearn import plotting
from Feature_Extraction.Shen_features.Classification_feature import atlas_path

reho_sub_1 = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-01/results/ReHo.nii.gz'
reho_sub_2 = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD/sub-02/results/ReHo.nii.gz'

mean_reho = mean_img(imgs=[reho_sub_1, reho_sub_2], n_jobs=2)

result = image.load_img(mean_reho)

shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True, strategy='mean',
                                          resampling_target="labels")

reho_rbd_data = shen_atlas.fit_transform(result)

rbd_img_masked = shen_atlas.inverse_transform(reho_rbd_data)

plotting.plot_stat_map(rbd_img_masked)
plotting.show()
