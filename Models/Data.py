import pandas as pd
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import chain
import numpy as np

data = pd.DataFrame(pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features'))

data.loc[:, 'FC'] = data['FC'].apply(ast.literal_eval)

sample = []

for k in range(len(data['FC'])):
    sample_list = []
    for j in range(len(data['FC'][k])):
        for u in data['FC'][k][j]:
            sample_list.append(u)

    sample_list = list(sample_list)
    sample.append(sample_list)

data.loc[:, 'Modified_FC'] = sample

data.reindex(columns=['FC', 'Modified_FC', 'REHO', 'ALFF', 'STATUS'])

data = data[['Modified_FC', 'REHO', 'ALFF', 'STATUS']]

data_path = os.path.join('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_data.csv')

data.to_csv(data_path, index=False)
