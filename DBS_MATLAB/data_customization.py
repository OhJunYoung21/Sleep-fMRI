import numpy as np
import pandas as pd
import os
from scipy.io import savemat

label_excel = pd.read_excel('/Users/oj/Desktop/shen_nodes_Broamann_label.xlsx')

labels = label_excel['anatomical']

print(labels.shape)

data = pd.read_pickle('DBS_shen_data.pkl')


savemat('NML_RBD_FC_test_1'
        '.mat', {"aa": data['STATUS'].tolist(), "s_all": data['FC'].tolist(),
                           "roi_names": labels.tolist()})

