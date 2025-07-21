import pandas as pd
import numpy as np
from nilearn import image, plotting
from nilearn import input_data, datasets, maskers
from nilearn.connectome import ConnectivityMeasure

sample = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Static_Feature_Extraction/Shen_features/FC_18_parameters.pkl'
)

print(sample.columns)
