import pandas as pd
import numpy as np
from nilearn import image, plotting
from nilearn import input_data, datasets, maskers
from nilearn.connectome import ConnectivityMeasure

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen = image.load_img(atlas_path)


def FC_for_shen(path):
    shen_masker = input_data.NiftiLabelsMasker(labels_img=shen, standardize=True, strategy='mean', t_r=3, low_pass=0.1,
                                               detrend=True,
                                               high_pass=0.01,
                                               resampling_target="labels")
    data = image.load_img(path)

    time_series = shen_masker.fit_transform(data)

    print(time_series.shape)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])

    return correlation_matrix


print(np.round(
    FC_for_shen('/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_confound_regressed_res_2/sub-01_confounds_regressed.nii.gz'),
    3))
