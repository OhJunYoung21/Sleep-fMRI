import os
import glob
import shutil

base_dir = "/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_post_prep"  # 여기에 대상 디렉토리 경로 입력

# 'sub-*' 패턴을 가진 폴더 찾기
sub_dirs = sorted(glob.glob(os.path.join(base_dir, "sub-*", "func")))

target_path = '/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_mask'

for sub_dir in sub_dirs:
    # 'desc_brain_mask' 파일 경로 찾기
    mask_path = glob.glob(os.path.join(sub_dir,
                                       'sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')) + glob.glob(
        os.path.join(sub_dir,
                     'sub-*_task-BRAINMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')) + glob.glob(
        os.path.join(sub_dir,
                     'sub-*_task-DIFFUSIONMRINONCONTRAST_acq-WIPfMRIRESTCLEAR_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')) + glob.glob(
        os.path.join(sub_dir,
                     'sub-*_task-BRAINMRINONCONTRAST_acq-fMRIRESTSENSE_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')) + glob.glob(
        os.path.join(sub_dir,
                     'sub-*_task-BRAINMRINONCONTRAST_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')) + glob.glob(
        os.path.join(sub_dir,
                     'sub-*_task-RESEARCHMRI_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))

    if not mask_path:
        print("empty")
    else:
        shutil.move(mask_path[0], target_path)
