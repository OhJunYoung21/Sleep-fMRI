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

NML_RBD_pkl = pd.read_pickle('../Statistic/NML_RBD_data.pkl')

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


def non_feature_selected_RF(feature_name: str):
    accuracy_score_mean = []
    f1_score_mean = []
    precision_mean = []
    recall_mean = []
    feature_difference = []

    SVM_result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

    ### selected_data_1 contains RBD data, selected_data_0 contains NML data

    selected_data_1 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    rkf_split_1 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)
    rkf_split_0 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)

    i = 0

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

        train_data[feature_name] = [item for item in train_data[feature_name]]
        test_data[feature_name] = [item for item in test_data[feature_name]]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        rf.fit(np.array(train_data[feature_name].tolist()), train_data['STATUS'])

        y_pred = rf.predict(np.array(test_data[feature_name].tolist()))

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

    SVM_result['Accuracy'] = np.round(np.mean(accuracy_score_mean), 2)
    SVM_result['Precision'] = np.round(np.mean(precision_mean), 2)
    SVM_result['Recall'] = np.round(np.mean(recall_mean), 2)
    SVM_result['F1'] = (np.round(np.mean(f1_score_mean), 2))

    pd.DataFrame(SVM_result, index=[1]).to_excel(f'./Results/RF/Non_feature_selected_RF/RF_{feature_name}_result.xlsx')

    return


def feature_selected_RF(feature_name: str, p_value: str):
    accuracy_score_mean = []
    f1_score_mean = []
    precision_mean = []
    recall_mean = []

    SVM_result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

    selected_nodes = \
        pd.read_csv(
            f'../Statistic/statistic_result_table/{feature_name}/{feature_name}_result_final_p_value_{p_value}.csv')[
            'Feature_Index'] - 1

    ### selected_data_1 contains RBD data, selected_data_0 contains NML data

    selected_data_1 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    selected_data_1[feature_name] = selected_data_1[feature_name].apply(lambda x: [x[i] for i in selected_nodes])
    selected_data_0[feature_name] = selected_data_0[feature_name].apply(lambda x: [x[i] for i in selected_nodes])

    rkf_split_1 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)
    rkf_split_0 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)

    i = 0

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

        train_data[feature_name] = [item for item in train_data[feature_name]]
        test_data[feature_name] = [item for item in test_data[feature_name]]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        rf.fit(np.array(train_data[feature_name].tolist()), train_data['STATUS'])

        y_pred = rf.predict(np.array(test_data[feature_name].tolist()))

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

    SVM_result['Accuracy'] = np.round(np.mean(accuracy_score_mean), 2)
    SVM_result['Precision'] = np.round(np.mean(precision_mean), 2)
    SVM_result['Recall'] = np.round(np.mean(recall_mean), 2)
    SVM_result['F1'] = (np.round(np.mean(f1_score_mean), 2))

    pd.DataFrame(SVM_result, index=[1]).to_excel(
        f'./Results/RF/Statistic_feature_selected_RF/RF_{feature_name}_result_{p_value}.xlsx')

    return SVM_result


feature_selected_RF("ALFF", "0.05")
feature_selected_RF("ALFF", "0.01")
feature_selected_RF("fALFF", "0.05")
feature_selected_RF("fALFF", "0.01")
feature_selected_RF("REHO", "0.05")
feature_selected_RF("REHO", "0.01")
feature_selected_RF("FC", "0.05")
feature_selected_RF("FC", "0.01")
feature_selected_RF("FC", "0.001")
