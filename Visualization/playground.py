import pandas as pd

shen_static_pkl = pd.read_pickle('../Static_Feature_Extraction/Shen_features/shen_268_CNN.pkl')

print(shen_static_pkl['FC'][0].shape)
