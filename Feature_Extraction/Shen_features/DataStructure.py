import re

import pandas as pd
import os
import numpy as np
import glob
from Feature_Extraction import Shen_features
import Classification_feature
from Classification_feature import calculate_3dReHo
from Classification_feature import region_reho_average
from Classification_feature import atlas_path, FC_extraction

shen_data = pd.DataFrame(index=None)

shen_data['FC'] = None
shen_data['ALFF'] = None
shen_data['REHO'] = None
shen_data['STATUS'] = None

ReHo = []
FC = []

root_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD'

reho_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho'

files_rbd = glob.glob(os.path.join(root_dir, 'sub-*_confounds_regressed.nii.gz'))

files_rbd = sorted(files_rbd)


## 데이터프레임안의 요소들을 전부 지우는 함수이다. 혹시나 데이터프레임안의 데이터가 꼬이는 경우에 빠른 초기화를 위해 제작하였다.
def delete():
    shen_data.iloc[:, :] = None
    return shen_data


### reho를 계산해서 reho 디렉토리 안에 넣어주는 코드

def input_reho():
    for file in files_rbd:
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', file)

        if match:
            extracted_part = match.group(1)

        calculate_3dReHo(file, extracted_part)

    return


def input_reho_shen():
    reho_path = glob.glob(os.path.join(reho_dir, 'reho_*.nii.gz'))

    for file in reho_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_reho = region_reho_average(file, atlas_path)
        ReHo.append(shen_reho)
    return


def input_fc_shen():
    for file in files_rbd:
        correlation_matrix = FC_extraction(file, atlas_path)

        correlation_matrix = correlation_matrix.tolist()

        for j in range(len(correlation_matrix)):
            correlation_matrix[j][:] = correlation_matrix[j][:-(len(correlation_matrix) - j)]

        FC.append(correlation_matrix)

    return


input_fc_shen()

print(len(FC[0]))

'''
for k in range(len(ReHo)):
    shen_data.loc[k] = [FC[k], '', ReHo[k], 1]
'''
