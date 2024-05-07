## fMRIprep output을 사용해서 noise regression 하기


### confound_timeseries파일 불러오기

~~~python3
import nilearn

import pandas as pd

confound = pd.read_table("/Users/ojun-yong/Desktop/bids_ex/fmriprep/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv")
~~~

### 파일에 있는 결측치(Nan)를 0으로 대체하기

~~~python3
import numpy as np

confounds = confounds.fillna(0)
confounds.head()
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

### 상관 매트릭스(상관 행렬) 계산하고 시각화하기

~~~python3
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

from nilearn import plotting

# Make a large figure
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)
# The labels we have start with the background (0), hence we skip the
# first label
# matrices are ordered for block-like representation
plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="Confounds",
    reorder=True,
)
~~~

