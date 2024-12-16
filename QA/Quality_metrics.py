import os
import pandas as pd

confound_json = pd.read_json(
    "/Users/oj/Desktop/Yoo_Lab/post_fMRI/post_prep_HC/sub-01/func/sub-01_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.json")
normalization_json = pd.read_json(
    "/Users/oj/Desktop/Yoo_Lab/post_fMRI/post_prep_HC/sub-01/func/sub-01_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.json")

print(confound_json.columns)
