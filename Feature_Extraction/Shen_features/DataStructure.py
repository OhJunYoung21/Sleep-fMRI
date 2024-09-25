import re

import pandas as pd
import os
import numpy as np
import glob
from Feature_Extraction import Shen_features
import Feature_Extraction.Shen_features.Classification_feature
from Feature_Extraction.Shen_features.Classification_feature import calculate_Bandpass_HC, calculate_Bandpass_RBD
from Classification_feature import calculate_3dReHo_HC, calculate_3dReHo_RBD
from Classification_feature import region_reho_average
from Classification_feature import atlas_path, FC_extraction
from Classification_feature import region_alff_average
from typing import List

shen_data = pd.DataFrame(index=None)

shen_data['FC'] = None
shen_data['ALFF'] = None
shen_data['REHO'] = None
shen_data['STATUS'] = None

ReHo_RBD = []
FC_RBD = []
ALFF_RBD = []

ReHo_HC = []
FC_HC = []
ALFF_HC = []

root_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD'

root_hc_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC'

reho_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho'

alff_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff'

reho_hc_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/reho'

alff_hc_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/alff'

files_rbd = glob.glob(os.path.join(root_rbd_dir, 'sub-*_confounds_regressed.nii.gz'))

files_hc = glob.glob(os.path.join(root_hc_dir, 'sub-*_confounds_regressed.nii.gz'))
'''
files_rbd = sorted(files_rbd)

files_hc = sorted(files_hc)
'''


## 데이터프레임안의 요소들을 전부 지우는 함수이다. 혹시나 데이터프레임안의 데이터가 꼬이는 경우에 빠른 초기화를 위해 제작하였다.
def delete():
    shen_data.iloc[:, :] = None
    return shen_data


### reho를 계산해서 reho 디렉토리 안에 넣어주는 코드

def input_reho_RBD(files_path: str):
    for file in files_rbd:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_3dReHo_RBD(file, extracted_part)

    return


def input_reho_HC(files_path: str):
    for file in files_rbd:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_3dReHo_HC(file, extracted_part)

    return


### input_alff()는 alff파일을 만들어서 로컬에 저장하는 함수입니다.

def input_alff_RBD(files_path: str):
    for file in files_path:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_Bandpass_RBD(file, extracted_part)

    return


def input_alff_HC(files_path: str):
    for file in files_path:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_Bandpass_HC(file, extracted_part)

    return


def input_reho_shen(data: List):
    reho_path = glob.glob(os.path.join(reho_rbd_dir, 'reho_*.nii.gz'))

    for file in reho_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_reho = region_reho_average(file, atlas_path)
        data.append(shen_reho)
    return


def input_fc_shen(data: List):
    for file in files_rbd:
        correlation_matrix = FC_extraction(file, atlas_path)

        correlation_matrix = correlation_matrix.tolist()

        for j in range(len(correlation_matrix)):
            correlation_matrix[j][:] = correlation_matrix[j][:-(len(correlation_matrix) - j)]

        data.append(correlation_matrix)
    return


### 로컬의 alff폴더에서 파일을 읽어온 후, shen_atlas를 적용하고 ALFF 리스트에 넣어준다.
def input_alff_shen(data: List):
    alff_path = glob.glob(os.path.join(alff_rbd_dir, 'alff_transformed_*.nii.gz'))

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_alff = region_alff_average(file, atlas_path)
        data.append(shen_alff)
    return


input_alff_HC(files_hc)
