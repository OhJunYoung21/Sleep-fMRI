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

### Masker를 설정하고, fit_transform함수를 사용해서 뇌의 각 영역별 noise regressed된 signal추출하기

~~~python3
from nilearn.maskers import NiftiLabelsMasker

masker = NiftiLabelsMasker(
    labels_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(file_name, confounds=confounds)
~~~
