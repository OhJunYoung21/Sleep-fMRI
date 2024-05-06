## fMRIprep output을 사용해서 noise regression 하기


### confound_timeseries파일 불러오기

~~~python3
import nilearn

import pandas as pd

confound = pd.read_table("/Users/ojun-yong/Desktop/bids_ex/fmriprep/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv")
~~~

### 파일 확인하기

~~~python3
confound.head()
~~~

### Confound 정의하기

~~~python3
from nilearn.interfaces.fmriprep import load_confounds

confounds_simple, sample_mask = load_confounds(
    file_name,
    strategy=["high_pass", "motion", "wm_csf"],
    motion="basic",
    wm_csf="basic",
)

print("The shape of the confounds matrix is:", confounds_simple.shape)
print(confounds_simple.columns)
print(confounds_simple.head())
~~~
