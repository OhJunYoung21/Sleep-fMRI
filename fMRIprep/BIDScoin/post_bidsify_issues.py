import pandas as pd
import glob
import os
import re
from nilearn import input_data
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import concat_imgs

### bidsify 실행 후 오류가 발생하면 한가지의 .nii.gz파일이 아닌 여러개의 .nii.gz파일이 생성된다. 이는 각 TR마다 촬영된 3D image가 전체 timepoint만큼 생성된 것으로, 4D이미지를 원하는 우리는 이 3D image를 4D로 합쳐줘야 한다.

### 해당 과정은 nilearn이 제공하는 concat_image를 써서 해결하도록 한다.

### 1. subject의 func 들을 순회한다.
### 2. subject내의 func 파일에 여러개의 .nii.gz,json파일이 들어있는 경우를 if문으로 처리한다.
### 3. .json파일 하나를


nifti_files = glob.glob(
    '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS/sub-04/func/sub-04_task-RESEARCHMRI_acq-AxialfMRIrest*_bold.nii.gz')

json_files = glob.glob(
    '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS/sub-04/func/sub-04_task-RESEARCHMRI_acq-AxialfMRIrest*_bold.json')


def extract_t_number(file):
    match = re.search(r't(\d+)_bold', file)  # Find the number after 't'
    return int(match.group(1)) if match else int(0)


sorted_file = sorted(nifti_files, key=extract_t_number)

data = concat_imgs(sorted_file)

for file in nifti_files:
    try:
        os.remove(file)
    except:
        print("error")

for file in json_files:
    try:
        if file.endswith('rest_bold.json'):
            continue
        else:
            os.remove(file)
    except:
        print("error")

output_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS/sub-04/func/sub-04_task-RESEARCHMRI_acq-AxialfMRIrest_bold.nii.gz'

data.to_filename(output_path)
