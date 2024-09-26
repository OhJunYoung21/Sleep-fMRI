import re

import pandas as pd
import os
import numpy as np
import glob
from Feature_Extraction import Shen_features
import Feature_Extraction.Shen_features.Classification_feature
from Classification_feature import calculate_Bandpass
from Classification_feature import calculate_3dReHo
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

feature_path = '/Users/oj/Desktop/Yoo_Lab/Classification_Features'


## 데이터프레임안의 요소들을 전부 지우는 함수이다. 혹시나 데이터프레임안의 데이터가 꼬이는 경우에 빠른 초기화를 위해 제작하였다.
def delete():
    shen_data.iloc[:, :] = None
    return shen_data


### reho를 계산해서 reho 디렉토리 안에 넣어주는 코드

def input_fc(files_path: str, data: List):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        connectivity = FC_extraction(file, atlas_path)
        connectivity = connectivity.tolist()

        for j in range(len(connectivity)):
            connectivity[j][:] = connectivity[j][:-(len(connectivity) - j)]

        data.append(connectivity)

    return


def input_reho(files_path: str):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_3dReHo(file, extracted_part, os.path.join(files_path + '/reho'))

    return


### input_alff()는 alff파일을 만들어서 로컬에 저장하는 함수입니다.

def input_alff(files_path: str):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_Bandpass(file, extracted_part, os.path.join(files_path + '/alff'))

    return


def input_reho_shen(file_path: str, data: List):
    reho_path = glob.glob(os.path.join(file_path, 'reho_*.nii.gz'))

    for file in reho_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_reho = region_reho_average(file, atlas_path)
        data.append(shen_reho)
    return


### 로컬의 alff폴더에서 파일을 읽어온 후, shen_atlas를 적용하고 ALFF 리스트에 넣어준다.
def input_alff_shen(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'alff_transformed_*.nii.gz'))

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_alff = region_alff_average(file, atlas_path)
        data.append(shen_alff)
    return


input_reho_shen(reho_hc_dir, ReHo_HC)
input_alff_shen(alff_hc_dir, ALFF_HC)
input_fc(root_hc_dir, FC_HC)
input_reho_shen(reho_rbd_dir, ReHo_RBD)
input_alff_shen(alff_rbd_dir, ALFF_RBD)
input_fc(root_rbd_dir, FC_RBD)

len_hc = len(ReHo_HC)
len_rbd = len(ReHo_RBD)

for j in range(len_hc):
    shen_data.loc[j] = [FC_HC[j], ALFF_HC[j], ReHo_HC[j], 0]

for k in range(len_rbd):
    shen_data.loc[len_hc + k] = [FC_RBD[k], ALFF_RBD[k], ReHo_RBD[k], 1]

shen_data_path = os.path.join(feature_path, 'Shen/Shen_features')

shen_data.to_csv(shen_data_path, index=False)
