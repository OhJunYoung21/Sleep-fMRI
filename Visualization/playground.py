import pandas as pd

BASC_pkl = pd.read_pickle('../Feature_Extraction/BASC_features/non_PCA_features/basc_325_non_PCA.pkl')

print(len(BASC_pkl['REHO'][0][0]))
