import re
import pandas as pd
import os
import numpy as np
import glob
from typing import List
from CPAC import alff
from Comparison_features.rsfmri import static_measures, dynamic_measures
from Dynamic_Feature_Extraction.Shen_features.Shen_features import FC_extraction, shen_alff_average, \
    shen_falff_average, shen_reho_average

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

CPAC_rbd = '/Users/oj/Desktop/Yoo_Lab/CPAC/dynamic/RBD'

CPAC_hc = '/Users/oj/Desktop/Yoo_Lab/CPAC/dynamic/HC'

CPAC_hc_dynamic = 'Users/oj/Desktop/Yoo_Lab/CPAC/dynamic/HC'
CPAC_rbd_dynamic = 'Users/oj/Desktop/Yoo_Lab/CPAC/dynamic/RBD'

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
        connectivity = FC_extraction(file)

        connectivity = np.array(connectivity)

        '''
        connectivity = (connectivity + connectivity.T) / 2  # 대칭화
        np.fill_diagonal(connectivity
                         , 0)

        vectorized_fc = connectivity[np.triu_indices(268, k=1)]
        '''

        data.append(connectivity)

    return data


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


def input_dynamic_features(files_path: str, mask_path: str, status: str):
    fmri_files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))
    mask_files = glob.glob(os.path.join(mask_path, 'sub-*_desc-brain_mask.nii.gz'))

    fmri_files = sorted(fmri_files)
    mask_files = sorted(mask_files)

    for idx in range(len(fmri_files)):
        match = re.search(r'sub-(.*)_confounds_regressed.nii.gz', fmri_files[idx])

        if match:
            extracted_part = match.group(1)

        output_path = os.path.join(f'/Users/oj/Desktop/Yoo_Lab/CPAC/dynamic/{status}/sub-{extracted_part}')

        dynamic_measures(fmri_files[idx], mask_files[idx], output_path,
                         nClusterSize=27, nJobs=1)

    return


def make_reho_shen(file_path: str, data: List):
    reho_path = glob.glob(os.path.join(file_path, 'sub-*/results/ReHo_merged.nii.gz'))

    for file in reho_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_reho = shen_reho_average(file)
        data.append(shen_reho)
    return


### 로컬의 alff폴더에서 파일을 읽어온 후, shen_atlas를 적용하고 ALFF 리스트에 넣어준다.
def make_alff_shen(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/alff_merged.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_alff = shen_alff_average(file)
        data.append(shen_alff)
    return


def make_falff_shen(file_path: str, data: List):
    falff_path = glob.glob(os.path.join(file_path, 'sub-*/results/falff_merged.nii.gz'))

    falff_path = sorted(falff_path)

    for file in falff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        shen_falff = shen_falff_average(file)
        data.append(shen_falff)
    return


input_fc(root_rbd_dir, FC_RBD)
input_fc(root_hc_dir, FC_HC)

make_reho_shen(CPAC_hc, ReHo_HC)
make_alff_shen(CPAC_hc, ALFF_HC)
make_falff_shen(CPAC_hc, fALFF_HC)

make_reho_shen(CPAC_rbd, ReHo_RBD)
make_alff_shen(CPAC_rbd, ALFF_RBD)
make_falff_shen(CPAC_rbd, fALFF_RBD)

len_hc = len(FC_HC)
len_rbd = len(FC_RBD)

for j in range(len_hc):
    shen_data.loc[j] = [FC_HC[j], ALFF_HC[j], ReHo_HC[j], fALFF_HC[j], 0]

for k in range(len_rbd):
    shen_data.loc[len_hc + k] = [FC_RBD[k], ALFF_RBD[k], ReHo_RBD[k], fALFF_RBD[k], 1]

shen_data.to_pickle('shen_268_dynamic.pkl')
