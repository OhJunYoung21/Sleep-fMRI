import re
import pandas as pd
import os
import numpy as np
import glob
from Static_Feature_Extraction.Schaefer_features.fit_features_to_atlas import fit_reho_atlas, fit_alff_atlas, \
    fit_FC_atlas
from typing import List
from CPAC import alff
from sklearn.decomposition import PCA
from scipy.stats import zscore
from Comparison_features.rsfmri import static_measures

covariate_RBD = pd.read_excel('/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/Data_covariate.xlsx', sheet_name='RBD',
                              header=1)

covariate_RBD = covariate_RBD.drop(columns=['Unnamed: 0', 'Unnamed: 4'])

covariate_RBD = covariate_RBD.dropna(subset=['sub-number', 'age', 'sex'])

covariate_RBD = covariate_RBD[covariate_RBD['sub-number'] != 'sub-17']

covariate_RBD = covariate_RBD[covariate_RBD['sub-number'] != 'sub-61']

covariate_RBD = covariate_RBD[covariate_RBD['sub-number'] != 'sub-98']

covariate_RBD = covariate_RBD[covariate_RBD['sub-number'] != 'sub-115']



'''

data = pd.read_pickle('./schaefer_data_RBD.pkl')

data = pd.concat([data, covariate_RBD], axis=1)

data.to_pickle('./schaefer_covariate_RBD.pkl')

schaefer_data_HC = pd.DataFrame()

schaefer_data_HC['FC'] = None
schaefer_data_HC['ALFF'] = None
schaefer_data_HC['REHO'] = None
schaefer_data_HC['fALFF'] = None
schaefer_data_HC['STATUS'] = None

ReHo_HC = []
FC_HC = []
ALFF_HC = []
fALFF_HC = []

'''

schaefer_data_RBD = pd.DataFrame()

schaefer_data_RBD['FC'] = None
schaefer_data_RBD['ALFF'] = None
schaefer_data_RBD['REHO'] = None
schaefer_data_RBD['fALFF'] = None
schaefer_data_RBD['STATUS'] = None

ReHo_RBD = []
FC_RBD = []
ALFF_RBD = []
fALFF_RBD = []

confounds_hc_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_confound_regressed'
mask_hc_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_mask'

confounds_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_confound_regressed'
mask_rbd_dir = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_mask'

CPAC_hc = '/Users/oj/Desktop/Yoo_Lab/CPAC/NML'
CPAC_rbd = '/Users/oj/Desktop/Yoo_Lab/CPAC/RBD'


def fc_for_data(files_path: str, data: List):
    files = glob.glob(os.path.join(files_path, 'sub-*_confounds_regressed.nii.gz'))

    files = sorted(files)

    for file in files:
        connectivity = fit_FC_atlas(file)

        connectivity = np.array(connectivity)[0]
        np.fill_diagonal(connectivity
                         , 0)

        vectorized_fc = connectivity[np.tril_indices(300, k=-1)]

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

    output_path = os.path.join(f'/Users/oj/Desktop/Yoo_Lab/CPAC/{status}/sub-{extracted_part}')

    static_measures(fmri_files[idx], mask_files[idx], output_path,
                    nClusterSize=27, nJobs=1)

    return


def reho_for_data(file_path: str, data: List):
    reho_path = glob.glob(os.path.join(file_path, 'sub-*/results/ReHo.nii.gz'))

    for file in reho_path:
        reho_atlas = fit_reho_atlas(file)

        data.append(reho_atlas)

    return


def alff_for_data(file_path: str, data: List):
    alff_path = glob.glob(os.path.join(file_path, 'sub-*/results/alff.nii.gz'))

    alff_path = sorted(alff_path)

    for file in alff_path:
        alff_atlas = fit_alff_atlas(file)
        data.append(alff_atlas)
    return


def falff_for_data(file_path: str, data: List):
    falff_path = glob.glob(os.path.join(file_path, 'sub-*/results/falff.nii.gz'))

    falff_path = sorted(falff_path)

    for file in falff_path:
        falff_atlas = fit_alff_atlas(file)
        data.append(falff_atlas)
    return


'''
result_hc = fc_for_data(confounds_hc_dir, FC_HC)

reho_for_data(CPAC_hc, ReHo_HC)
alff_for_data(CPAC_hc, ALFF_HC)
falff_for_data(CPAC_hc, fALFF_HC)

ALFF_HC = [k.tolist()[0] for k in ALFF_HC]
fALFF_HC = [k.tolist()[0] for k in fALFF_HC]
ReHo_HC = [k.tolist()[0] for k in ReHo_HC]

len_hc = len(result_hc)

for j in range(len_hc):
    schaefer_data_HC.loc[j] = [result_hc[j], ALFF_HC[j], ReHo_HC[j], fALFF_HC[j], 0]

schaefer_data_HC.to_pickle('./schaefer_data_HC.pkl')
'''

result_rbd = fc_for_data(confounds_rbd_dir, FC_RBD)

print(len(result_rbd))

reho_for_data(CPAC_rbd, ReHo_RBD)
alff_for_data(CPAC_rbd, ALFF_RBD)
falff_for_data(CPAC_rbd, fALFF_RBD)

ALFF_RBD = [k.tolist()[0] for k in ALFF_RBD]
fALFF_RBD = [k.tolist()[0] for k in fALFF_RBD]
ReHo_RBD = [k.tolist()[0] for k in ReHo_RBD]

len_rbd = len(result_rbd)

for k in range(len_rbd):
    schaefer_data_RBD.loc[k] = [result_rbd[k], ALFF_RBD[k], ReHo_RBD[k], fALFF_RBD[k], 1]

schaefer_data_RBD.to_pickle('./schaefer_data_RBD.pkl')

print(schaefer_data_RBD.shape)
