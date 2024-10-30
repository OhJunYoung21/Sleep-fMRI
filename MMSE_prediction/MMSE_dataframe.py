import pandas as pd
import ast
import os
from nilearn import plotting

'''
Schaefer_features = pd.read_csv(
    '/Users/oj/Desktop/Yoo_Lab/Classification_Features/MMSE/MMSE_regression_final.csv',
    converters={
        'FC': ast.literal_eval,
        'ALFF': ast.literal_eval,
        'fALFF': ast.literal_eval,
        'REHO': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)

MMSE_features = pd.read_excel('/Users/oj/Desktop/Yoo_Lab/Classification_Features/fMRI_RBD_NML_20240308_1030.xlsx')

MMSE_data = pd.DataFrame()

MMSE_data['FC'] = Schaefer_features['FC']

print(len(MMSE_features['MMSE']))
'''

data = pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features_final.csv')
print(data.head())
