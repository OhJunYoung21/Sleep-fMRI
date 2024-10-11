import ast
import pandas as pd
import nilearn
from nilearn import plotting
from nilearn import input_data
import os
from nilearn import image
from Feature_Extraction.Shen_features.Classification_feature import FC_extraction, file_path, atlas_path

correlation_matrix = FC_extraction(file_path, atlas_path)

plotting.plot_matrix(correlation_matrix,
                     title='Correlation Matrix',
                     labels=None,
                     tri='full',
                     colorbar=True,
                     vmax=1.0,
                     vmin=-1.0
                     )
plotting.show()
