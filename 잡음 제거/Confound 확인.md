### Confound 확인

fmriprep결과, confound들이 도출되는데, 이를 파이썬 코드를 사용하여 간단하게 확인할 수 있다.

~~~python3
import pandas as pd

file_path = "/Users/ojun-yong/Desktop/bids_ex/fmriprep/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv"
conf = pd.read_csv(file_path,sep='\t')
conf.head()
~~~
이렇게 하면, confound들의 종류와 그에 해당하는 값을 알 수 있다.

