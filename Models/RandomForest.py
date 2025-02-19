import pandas as pd
import ast
import os
from nilearn import plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import itertools
from collections import Counter

PET_pkl = pd.read_pickle('../PET_classification/PET_shen_static.pkl')

accuracy_score_mean = []
f1_score_mean = []
precision_mean = []
recall_mean = []
feature_difference = []


def avoid_duplication(nested_list):
    unique_elements = set()
    for sublist in nested_list:
        unique_elements.update(sublist)

    # 집합을 다시 리스트로 변환
    result = list(unique_elements)

    return result


accuracy_score_mean = []
feature_difference = []

feature_name = 'FC'

status_1_data = PET_pkl[PET_pkl['STATUS'] == 1]
status_0_data = PET_pkl[PET_pkl['STATUS'] == 0]
# Select only the REHO and STATUS columns
selected_data_1 = status_1_data[[feature_name, 'STATUS']]
selected_data_0 = status_0_data[[feature_name, 'STATUS']]

i = 0

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

PET_pkl['FC'] = PET_pkl['FC'].apply(lambda x: x.tolist())

feature_importance_list = []

for train_idx, test_idx in skf.split(PET_pkl[feature_name], PET_pkl['STATUS']):
    X_train, X_test = PET_pkl[feature_name].iloc[train_idx], PET_pkl[feature_name].iloc[test_idx]
    y_train, y_test = PET_pkl['STATUS'].iloc[train_idx], PET_pkl['STATUS'].iloc[test_idx]

    print(len(X_train), len(y_train))

    '''    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    feature_importance_list.append(rf.feature_importances_)
    '''

print(feature_importance_list)
