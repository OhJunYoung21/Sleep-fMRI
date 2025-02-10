s# functionalMRI

In this repository, I'll mainly deal with python3 codes processing fMRI nifti data.

## Data Preprocessing(데이터 전처리)

In this Directory, we'll mainly deal with fMRIprep's pipeline.

## Noise regression(데이터 분석)

In this Directory, I'll add some .ipnyb files performing noise regression. fMRIprep produce confounds_timeseries.tsv as its output. It is upto us how to deal with it.
Of course, It'll be greate if we could consider all of the confounds at once. However, if we do so, it'ss take tremendous amount of time to calculate. So, nilearn provides comfortable API choosing confounds for us. we only need to adjust few parameters!!😁
