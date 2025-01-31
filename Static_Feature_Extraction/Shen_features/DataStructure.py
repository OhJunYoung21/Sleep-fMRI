import re
import pandas as pd
import os
import numpy as np
import glob
from Static_Feature_Extraction.Shen_features.Classification_feature import region_reho_average, region_alff_average, \
    atlas_path, FC_extraction, local_connectivity
from typing import List
from CPAC import alff
from sklearn.decomposition import PCA
from scipy.stats import zscore
from Comparison_features.rsfmri import static_measures

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen_data = pd.DataFrame(index=None)

shen_data['FC'] = None
shen_data['ALFF'] = None
shen_data['REHO'] = None
shen_data['fALFF'] = None
shen_data['STATUS'] = None

ReHo_RBD = []
FC_RBD = []
FC_prior_RBD = []
ALFF_RBD = []
fALFF_RBD = []

ReHo_HC = []
FC_HC = []
FC_prior_HC = []
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

### CPAC에서는 mask파일이 있어야 alff,falff 그리고 reho와 같은 feature들을 추출해낼 수 있다.

mask_path_rbd = '/Users/oj/Desktop/Yoo_Lab/mask_rbd'
mask_path_hc = '/Users/oj/Desktop/Yoo_Lab/mask_hc'


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


def input_fc_selected(files_path: str, data: List):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        connectivity = FC_extraction(file)

        connectivity = np.array(connectivity)

        selected_regions = [1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 19, 21, 22, 30, 31, 41,
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

        connectivity = connectivity[0][np.ix_(selected_regions, selected_regions)]

        connectivity = (connectivity + connectivity.T) / 2  # 대칭화
        np.fill_diagonal(connectivity
                         , 0)

        vectorized_fc = connectivity[np.triu_indices(len(selected_regions), k=1)]

        data.append(vectorized_fc)

    return data


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
        ## region_alff_average는 268개의 alff값들을 반환한다.
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


result_hc = input_fc(root_hc_dir, FC_HC)
result_rbd = input_fc(root_rbd_dir, FC_RBD)

make_reho_shen(CPAC_hc, ReHo_HC)
make_alff_shen(CPAC_hc, ALFF_HC)
make_falff_shen(CPAC_hc, fALFF_HC)

make_reho_shen(CPAC_rbd, ReHo_RBD)
make_alff_shen(CPAC_rbd, ALFF_RBD)
make_falff_shen(CPAC_rbd, fALFF_RBD)

ALFF_RBD = [k.tolist()[0] for k in ALFF_RBD]
ALFF_HC = [k.tolist()[0] for k in ALFF_HC]
fALFF_RBD = [k.tolist()[0] for k in fALFF_RBD]
fALFF_HC = [k.tolist()[0] for k in fALFF_HC]
ReHo_RBD = [k.tolist()[0] for k in ReHo_RBD]
ReHo_HC = [k.tolist()[0] for k in ReHo_HC]

len_hc = len(result_hc)
len_rbd = len(result_rbd)

for j in range(len_rbd):
    shen_data.loc[j] = [result_rbd[j], ALFF_RBD[j], ReHo_RBD[j], fALFF_RBD[j], 1]

for k in range(len_hc):
    shen_data.loc[len_rbd + k] = [result_hc[k], ALFF_HC[k], ReHo_HC[k], fALFF_HC[k], 0]

### shen_data에서 각행의 alff,falff,reho값은 268 size의 리스트 혹은 numpy 로 구성되어 있다.

shen_data.to_pickle('shen_268_static.pkl')
