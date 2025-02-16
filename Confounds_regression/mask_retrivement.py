import os
import glob

base_dir = "/Users/oj/Desktop/Yoo_Lab/Yoo_data/RBD_PET_classification/RBD_PET_positive"  # 여기에 대상 디렉토리 경로 입력

# 'sub-*' 패턴을 가진 폴더 찾기
sub_dirs = sorted(glob.glob(os.path.join(base_dir, "sub-*", "func")))

# 각 sub-* 폴더에 접근하여 'desc_brain_mask' 파일 찾기
brain_mask_files = []

for sub_dir in sub_dirs:
    # 'desc_brain_mask' 파일 경로 찾기
    mask_path = glob.glob(
        os.path.join(sub_dir, "sub-*_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-brain_mask.nii.gz"))

    if not mask_path:
        print("empty")
    else:
        brain_mask_files.append(mask_path[0])

print(brain_mask_files)
