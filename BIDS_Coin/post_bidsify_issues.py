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

### file_name에는 여러개의 3D .nii.gz파일들이 들어있는 폴더를 지정한다.


nifti_files = glob.glob(
    '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_BIDS_negative/sub-16/func/sub-16_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR*_bold.nii.gz')

json_files = glob.glob(
    '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_BIDS_negative/sub-16/func/sub-16_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR*_bold.json')


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
        os.remove(file)
    except:
        print("error")

output_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_BIDS_negative/sub-16/func/sub-16_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_bold.nii.gz'

data.to_filename(output_path)
