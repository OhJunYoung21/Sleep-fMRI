import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import chain
import numpy as np

shen_features = pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features_final.csv',
                            converters={
                                'FC': ast.literal_eval,
                                'STATUS': ast.literal_eval
                            })

'''
alff_to_process = 'ALFF'  # 처리할 컬럼 이름
shen_features[alff_to_process] = shen_features[alff_to_process].str.split().str.join(',').flatten()

reho_to_process = 'REHO'  # 처리할 컬럼 이름
shen_features[reho_to_process] = shen_features[reho_to_process].str.split().str.join(',').flatten()

falff_to_process = 'fALFF'  # 처리할 컬럼 이름
shen_features[falff_to_process] = shen_features[falff_to_process].str.split().str.join(',').flatten()

shen_features.to_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features_final.csv', index=False)
'''

shen_features.to_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features_final_1.csv', index=False)
