import re
import pandas as pd
import os
import numpy as np
import glob
from Feature_Extraction.Shen_features.Classification_feature import calculate_Bandpass
from Feature_Extraction.Shen_features.Classification_feature import calculate_3dReHo
from Feature_Extraction.Shen_features.Classification_feature import region_reho_average
from Feature_Extraction.Shen_features.Classification_feature import atlas_path, FC_extraction
from Feature_Extraction.Shen_features.Classification_feature import region_alff_average
from typing import List
from CPAC import alff
from Comparison_features.rsfmri import static_measures

shen_data = pd.DataFrame(index=None)

shen_data['FC'] = None
shen_data['ALFF'] = None
shen_data['REHO'] = None
shen_data['fALFF'] = None
shen_data['STATUS'] = None

ReHo_RBD = []
FC_RBD = []
ALFF_RBD = []
fALFF_RBD = []

ReHo_HC = []
FC_HC = []
ALFF_HC = []
fALFF_HC = []

root_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD'

root_hc_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC'

reho_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/reho'

alff_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_RBD/alff'

reho_hc_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/reho'

alff_hc_dir = '/Users/oj/Desktop/Yoo_Lab/post_fMRI/confounds_regressed_HC/alff'

feature_path = '/Users/oj/Desktop/Yoo_Lab/Classification_Features'

CPAC_rbd = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD'

CPAC_hc = '/Users/oj/Desktop/Yoo_Lab/CPAC/HC'

mask_path_rbd = '/Users/oj/Desktop/Yoo_Lab/mask_rbd'
mask_path_hc = '/Users/oj/Desktop/Yoo_Lab/mask_hc'


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

        ''' 
        for j in range(len(connectivity)):
            connectivity[j][:] = connectivity[j][:-(len(connectivity) - j)]
        '''

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

def input_features(files_path: str, mask_path: str, status: str):
    fmri_files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))
    mask_files = glob.glob(os.path.join(mask_path, 'sub-*_desc-brain_mask.nii.gz'))

    fmri_files = sorted(fmri_files)
    mask_files = sorted(mask_files)

    for idx in range(len(fmri_files)):
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', fmri_files[idx])

        if match:
            extracted_part = match.group(1)

        output_path = os.path.join(f'/Users/oj/Desktop/Yoo_Lab/CPAC/{status}/sub-{extracted_part}')

        static_measures(fmri_files[idx], mask_files[idx], output_path,
                        nClusterSize=27, nJobs=1)

    return


def make_reho_shen(file_path: str, data: List):
    reho_path = glob.glob(os.path.join(file_path, 'sub-*/results/ReHo.nii.gz'))

    for file in reho_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_reho = region_reho_average(file, atlas_path)
        data.append(shen_reho)
    return


### 로컬의 alff폴더에서 파일을 읽어온 후, shen_atlas를 적용하고 ALFF 리스트에 넣어준다.
def make_alff_shen(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/alff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_alff = region_alff_average(file, atlas_path)
        data.append(shen_alff)
    return


def make_falff_shen(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/falff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_alff = region_alff_average(file, atlas_path)
        data.append(shen_alff)
    return


'''
input_features(root_rbd_dir, mask_path_rbd, "RBD")
input_features(root_hc_dir, mask_path_hc, "HC")
'''

make_reho_shen(CPAC_hc, ReHo_HC)
make_alff_shen(CPAC_hc, ALFF_HC)
make_falff_shen(CPAC_hc, fALFF_HC)
input_fc(root_hc_dir, FC_HC)

make_reho_shen(CPAC_rbd, ReHo_RBD)
make_alff_shen(CPAC_rbd, ALFF_RBD)
make_falff_shen(CPAC_rbd, fALFF_RBD)
input_fc(root_rbd_dir, FC_RBD)

len_hc = len(ReHo_HC)
len_rbd = len(ReHo_RBD)

for j in range(len_hc):
    shen_data.loc[j] = [FC_HC[j], ALFF_HC[j], ReHo_HC[j], fALFF_HC[j], 0]

for k in range(len_rbd):
    shen_data.loc[len_hc + k] = [FC_RBD[k], ALFF_RBD[k], ReHo_RBD[j], fALFF_RBD[j], 1]

shen_data_path = os.path.join(feature_path, 'Shen/Shen_features_ex')

shen_data.to_csv(shen_data_path, index=False)
