import pandas as pd
import glob
import os
from nilearn import input_data
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from Dynamic_Feature_Extraction.Shen_features.Shen_features import FC_extraction_time

file_path = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/sub-01_confounds_regressed.nii.gz'

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

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

shen_pkl = pd.read_pickle('../Dynamic_Feature_Extraction/Shen_features/shen_268_dynamic.pkl')

feature_nodes = [1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 19, 21, 22, 30, 31, 41,
                 43, 47, 48, 49, 50, 55, 56, 67, 69, 70, 71, 73, 74, 85, 86, 90, 96,
                 111, 112, 115, 116, 134, 138, 139, 141, 142, 143, 147, 154, 157,
                 164, 175, 177, 182, 184, 193, 196, 199, 200, 201, 203, 204, 206,
                 209, 210, 222, 223, 225, 227, 239, 240, 242, 246, 247]


def prior_extraction_connectivity(matrix, selected_regions):
    # 특정 region 간의 연결성 추출
    connectivity = matrix[0]

    connectivity = connectivity[selected_regions]

    '''
    # 대칭화
    connectivity = (connectivity + connectivity.T) / 2

    # 대각선 0 설정
    np.fill_diagonal(connectivity, 0)

    # 상삼각 행렬 벡터화
    vectorized_fc = connectivity[np.triu_indices(len(selected_regions), k=1)]
    '''

    return connectivity

