import netplotbrain
import pandas as pd
import numpy as np
import templateflow.api as tf
import itertools
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import center_of_mass
import os
import nilearn
from nilearn import image

atlas_path = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'  # 직접 다운로드한 경로 입력

pet_data = pd.read_pickle('../PET_classification/statistic_results/t_test_REHO.pkl')

target_location = (pet_data['Region'][pet_data['p-value'] < 0.05]).tolist()

# 2. Atlas 이미지 로드
img = image.load_img(atlas_path)
atlas_data = img.get_fdata()
affine = img.affine

# 3. ROI 레이블 목록 (0: background 제외)
roi_labels = np.unique(atlas_data)
roi_labels = roi_labels[roi_labels != 0]

# 4. 각 ROI의 중심 좌표 계산
centers = []
for roi in roi_labels:
    mask = atlas_data == roi
    com_voxel = center_of_mass(mask)
    com_mni = nib.affines.apply_affine(affine, com_voxel)  # voxel → MNI 좌표
    centers.append({
        'label': int(roi),
        'x': com_mni[0],
        'y': com_mni[1],
        'z': com_mni[2]
    })

# 5. DataFrame으로 저장
df_com = pd.DataFrame(centers)

netplotbrain.plot(nodes=df_com.iloc[target_location],

                  )
plt.show()
