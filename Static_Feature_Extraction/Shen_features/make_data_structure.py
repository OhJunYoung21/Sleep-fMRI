import re
import pandas as pd
import os
import numpy as np
import glob
from Static_Feature_Extraction.Shen_features.fit_features_to_atlas import reho_for_shen, alff_for_shen, falff_for_shen, \
    FC_for_shen
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

confounds_hc_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_confound_regressed_res_2_18_parameters'
mask_hc_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_mask_res_2'

confounds_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_confound_regressed_res_2_18_parameters'
mask_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_mask_res_2'

CPAC_hc = '/Users/oj/Desktop/Yoo_Lab/CPAC/HC_res_2'
CPAC_rbd = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD_res_2'


def input_fc(files_path: str, data: List):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        connectivity = FC_for_shen(file)

        connectivity = np.array(connectivity)[0]

        connectivity = (connectivity + connectivity.T) / 2  # 대칭화

        np.fill_diagonal(connectivity
                         , 0)

        vectorized_fc = connectivity[np.tril_indices(268, k=-1)]

        data.append(vectorized_fc)

    return data, print("FC complete!!!")


def input_fc_selected(files_path: str, data: List):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        connectivity = FC_for_shen(file)

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

        vectorized_fc = connectivity[np.triu_indices(len(selected_regions), k=-1)]

        data.append(vectorized_fc)

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

        output_path = os.path.join(f'/Users/oj/Desktop/Yoo_Lab/CPAC/{status}_res_2/sub-{extracted_part}')

        static_measures(fmri_files[idx], mask_files[idx], output_path,
                        nClusterSize=27, nJobs=1)

    return


input_features(confounds_rbd_dir, mask_rbd_dir, "RBD")


def reho_for_data(file_path: str, data: List):
    reho_path = glob.glob(os.path.join(file_path, 'sub-*/results/ReHo.nii.gz'))

    for file in reho_path:
        shen_reho = reho_for_shen(file)
        data.append(shen_reho)
    return


def alff_for_data(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/alff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        ## region_alff_average는 268개의 alff값들을 반환한다.
        shen_alff = falff_for_shen(file)
        data.append(shen_alff)
    return


def falff_for_data(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/falff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        shen_alff = falff_for_shen(file)
        data.append(shen_alff)
    return


'''
result_hc = input_fc(confounds_hc_dir, FC_HC)

reho_for_data(CPAC_hc, ReHo_HC)
alff_for_data(CPAC_hc, ALFF_HC)
falff_for_data(CPAC_hc, fALFF_HC)

ReHo_HC = [k.tolist()[0] for k in ReHo_HC]
ALFF_HC = [k.tolist()[0] for k in ALFF_HC]
fALFF_HC = [k.tolist()[0] for k in fALFF_HC]

print(len(ReHo_HC), len(ALFF_HC))

for k in range(len(result_hc)):
    shen_data.loc[k] = [result_hc[k], ALFF_HC[k], ReHo_HC[k], fALFF_HC[k], 0]

result_rbd = input_fc(confounds_rbd_dir, FC_RBD)

reho_for_data(CPAC_rbd, ReHo_RBD)
alff_for_data(CPAC_rbd, ALFF_RBD)
falff_for_data(CPAC_rbd, fALFF_RBD)

ALFF_RBD = [k.tolist()[0] for k in ALFF_RBD]
fALFF_RBD = [k.tolist()[0] for k in fALFF_RBD]
ReHo_RBD = [k.tolist()[0] for k in ReHo_RBD]

for j in range(len(result_rbd)):
    shen_data.loc[len(result_hc) + j] = [result_rbd[j], ALFF_RBD[j], ReHo_RBD[j], fALFF_RBD[j], 1]
'''
