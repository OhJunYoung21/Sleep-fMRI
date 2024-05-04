## Confound_timeseries.tsv파일을 업로드 하기

~~~python3
import nilearn

import pandas as pd

confound = pd.read_table("/Users/ojun-yong/Desktop/bids_ex/fmriprep/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv")
~~~
