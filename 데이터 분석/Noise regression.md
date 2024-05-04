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
