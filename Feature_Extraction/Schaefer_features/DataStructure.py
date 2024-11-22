import re
import pandas as pd
import os
import numpy as np
import glob
from Feature_Extraction.Schaefer_features.Classification_feature import region_reho_average, region_alff_average
from Feature_Extraction.Schaefer_features.Classification_feature import FC_extraction
import scipy.stats as stats
from typing import List
from CPAC import alff
from Comparison_features.rsfmri import static_measures
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io
from scipy.stats import zscore

schaefer_data = pd.DataFrame(index=None)

schaefer_data['FC'] = None
schaefer_data['ALFF'] = None
schaefer_data['REHO'] = None
schaefer_data['fALFF'] = None
schaefer_data['STATUS'] = None

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
    schaefer_data.iloc[:, :] = None
    return schaefer_data


'''
input_fc는 FC_extraction를 사용해서 추출한 Functional Connectivity를 PCA과정을 위해 벡터화 시켜주는 코드입니다.
'''


def input_fc(files_path: str, data: List):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        connectivity = FC_extraction(file)

        connectivity = np.array(connectivity)

        connectivity = (connectivity + connectivity.T) / 2  # 대칭화
        np.fill_diagonal(connectivity
                         , 0)

        vectorized_fc = connectivity[np.triu_indices(117, k=1)]

        data.append(vectorized_fc)

    return data


'''
input_features는 CPAC의 alff,reho등을 사용해서 주어진 file과 mask file을 사용해서 reho와 alff,falff를 추출한다.
'''


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


def make_reho_schaefer(file_path: str, data: List):
    reho_path = glob.glob(os.path.join(file_path, 'sub-*/results/ReHo.nii.gz'))

    for file in reho_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        schaefer_reho = region_reho_average(file)
        data.append(schaefer_reho)
    return


### 로컬의 alff폴더에서 파일을 읽어온 후, shen_atlas를 적용하고 ALFF 리스트에 넣어준다.
def make_alff_schaefer(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/alff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        schaefer_alff = region_alff_average(file)
        data.append(schaefer_alff)
    return


def make_falff_schaefer(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/falff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        ##Classification_feature에서 불러온 atlas_path를 매개변수로 넣어준다.
        schaefer_alff = region_alff_average(file)
        data.append(schaefer_alff)
    return


result_hc = input_fc(root_hc_dir, FC_HC)

result_rbd = input_fc(root_rbd_dir, FC_RBD)

result_pca = result_hc + result_rbd

pca = PCA(n_components=89)
result_pca = pca.fit_transform(result_pca)

FC_PCA_RBD_zscored = zscore(result_pca[:50], axis=0).tolist()
FC_PCA_HC_zscored = zscore(result_pca[50:], axis=0).tolist()

make_reho_schaefer(CPAC_hc, ReHo_HC)
make_alff_schaefer(CPAC_hc, ALFF_HC)
make_falff_schaefer(CPAC_hc, fALFF_HC)

make_reho_schaefer(CPAC_rbd, ReHo_RBD)
make_alff_schaefer(CPAC_rbd, ALFF_RBD)
make_falff_schaefer(CPAC_rbd, fALFF_RBD)

ALFF_RBD = [k.tolist()[0] for k in ALFF_RBD]
ALFF_HC = [k.tolist()[0] for k in ALFF_HC]
fALFF_RBD = [k.tolist()[0] for k in fALFF_RBD]
fALFF_HC = [k.tolist()[0] for k in fALFF_HC]
ReHo_RBD = [k.tolist()[0] for k in ReHo_RBD]
ReHo_HC = [k.tolist()[0] for k in ReHo_HC]

len_hc = len(FC_PCA_HC_zscored)
len_rbd = len(FC_PCA_RBD_zscored)

for j in range(len_rbd):
    schaefer_data.loc[j] = [FC_PCA_RBD_zscored[j], ALFF_RBD[j], ReHo_RBD[j], fALFF_RBD[j], 1]

for k in range(len_hc):
    schaefer_data.loc[len_rbd + k] = [FC_PCA_HC_zscored[k], ALFF_HC[k], ReHo_HC[k], fALFF_HC[k], 0]

schaefer_data.to_pickle('schaefer_200_pkl')
