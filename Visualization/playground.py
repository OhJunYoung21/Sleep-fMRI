import pandas as pd
import glob
import os
from nilearn import input_data
from nilearn import image
from nilearn.connectome import ConnectivityMeasure

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'
file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'
import numpy as np

feature_nodes = [1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 19, 21, 22, 30, 31, 41,
                 43, 47, 48,
                 49, 50, 55, 56, 67, 69, 70, 71, 73, 74, 85, 86, 90, 96,
                 111, 112, 115, 116, 134, 138
    , 139
    , 141
    , 142
    , 143
    , 147
    , 154
    , 157
    , 164
    , 175
    , 177
    , 182
    , 184
    , 193
    , 196
    , 199
    , 200
    , 201
    , 203
    , 204
    , 206
    , 209
    , 210
    , 222
    , 223
    , 225
    , 227
    , 239
    , 240
    , 242
    , 246
    , 247]


def FC_extraction(path):
    shen_atlas = input_data.NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    data = image.load_img(path)

    time_series = shen_atlas.fit_transform(data)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])

    matrix = correlation_matrix[0][np.ix_(feature_nodes, feature_nodes)]

    connectivity = (matrix + matrix.T) / 2  # 대칭화

    np.fill_diagonal(connectivity
                     , 0)

    vectorized_fc = connectivity[np.triu_indices(len(feature_nodes), k=1)]

    return vectorized_fc


print(len(FC_extraction(file_path)))
