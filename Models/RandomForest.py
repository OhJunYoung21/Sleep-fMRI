import pandas as pd
import ast
import os
from nilearn import plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

feature_name = 'fALFF'

status_1_data = PET_pkl[PET_pkl['STATUS'] == 1]
status_0_data = PET_pkl[PET_pkl['STATUS'] == 0]
# Select only the REHO and STATUS columns
selected_data_1 = status_1_data[[feature_name, 'STATUS']]
selected_data_0 = status_0_data[[feature_name, 'STATUS']]

i = 0

rkf_split_1 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)
rkf_split_0 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)

for (train_idx_1, test_idx_1), (train_idx_0, test_idx_0) in zip(
        rkf_split_1.split(selected_data_1),
        rkf_split_0.split(selected_data_0)):
    # 라벨 1 데이터의 훈련/테스트 분리
    train_1 = selected_data_1.iloc[train_idx_1]
    test_1 = selected_data_1.iloc[test_idx_1]

    # 라벨 0 데이터의 훈련/테스트 분리
    train_0 = selected_data_0.iloc[train_idx_0]
    test_0 = selected_data_0.iloc[test_idx_0]

    # 훈련 데이터와 테스트 데이터 결합

    train_data = pd.concat([train_1, train_0], axis=0).reset_index(drop=True)
    test_data = pd.concat([test_1, test_0], axis=0).reset_index(drop=True)

    '''
    train_data[feature_name] = [item[0] for item in train_data[feature_name]]
    test_data[feature_name] = [item[0] for item in test_data[feature_name]]
    '''

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(np.array(train_data[feature_name].tolist()), train_data['STATUS'])

    feature_importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature Index": np.arange(268),
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    feature_difference.append((feature_importance_df["Feature Index"][:10]).tolist())

    y_pred = model.predict(np.array(test_data[feature_name].tolist()))

    accuracy = accuracy_score(y_pred.tolist(), test_data['STATUS'])
    precision = precision_score(y_pred.tolist(), test_data['STATUS'])
    recall = recall_score(y_pred.tolist(), test_data['STATUS'])
    f1 = f1_score(y_pred.tolist(), test_data['STATUS'])

    i += 1

    print(f"{i}th accuracy : {accuracy:.2f}")

    accuracy_score_mean.append(accuracy)
    f1_score_mean.append(f1)
    precision_mean.append(precision)
    recall_mean.append(recall)

print(f"average of accuracy : {np.round(np.mean(accuracy_score_mean), 2)}")
print(f"average of f1-score : {np.round(np.mean(f1_score_mean), 2)}")
print(f"average of recall : {np.round(np.mean(recall_mean), 2)}")
print(f"average of precision : {np.round(np.mean(precision_mean), 2)}")

counter = Counter(list(itertools.chain(*feature_difference)))

top_10 = counter.most_common(10)

print(top_10)
