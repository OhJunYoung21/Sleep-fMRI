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
### 3.


def extract_sub_number(file):
    match = re.search(r'sub-(\d+)', file)  # Find the number after 't'
    return match.group(1) if match else int(0)


def extract_t_number(file):
    match = re.search(r't(\d+)_bold', file)  # Find the number after 't'
    return int(match.group(1)) if match else int(0)


root_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/NML_post_BIDS'


def concat_images(root: str):
    for file in sorted(os.listdir(root))[7:]:
        target_path = os.path.join(file, 'func')

        subject_number = extract_sub_number(target_path)

        try:
            ### /func 디렉토리 안의 파일들을 조회한다.(.nii.gz, .json)

            nifti_files = glob.glob(os.path.join(root, target_path,
                                                 'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest*_bold.nii.gz')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR*_bold.nii.gz')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR*_bold.nii.gz')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE*_bold.nii.gz')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest*_bold.nii.gz')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_rec-RESEARCHMRI*_bold.nii.gz')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest*_bold.nii.gz'))

            json_files = glob.glob(os.path.join(root, target_path,
                                                'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest*_bold.json')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR*_bold.json')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR*_bold.json')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE*_bold.json')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest*_bold.json')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_rec-RESEARCHMRI*_bold.json')) + glob.glob(
                os.path.join(root, target_path,
                             'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest*_bold.json'))

            if len(json_files) > 5:
                sorted_file_nifti = sorted(nifti_files, key=extract_t_number)
                sorted_file_json = sorted(nifti_files, key=extract_t_number)

                nifti_name = sorted_file_nifti[0]
                json_name = sorted_file_json[0]

                data = concat_imgs(sorted_file_nifti)

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

                output_path = os.path.join(root, target_path, 'func',
                                           nifti_name)

                data.to_filename(output_path)
            else:
                print('folder has 2 files')



        ### 만일, 조회한 파일들의 길이가 5를 넘기면, 이는 bold가 4d가 아니라는 뜻이므로 수정이 필요하다.

        except:
            print(f"{subject_number},error")

    return


concat_images(root_path)
