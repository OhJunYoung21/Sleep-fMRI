from random import shuffle

import pandas as pd
import ast
import os
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RepeatedKFold
from collections import Counter

from sklearn.svm import SVC

NML_RBD_pkl = pd.read_pickle('../Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_NML_RBD.pkl')


def modify_p_value(feature_name: str):
    data = pd.read_csv(
        f'../Statistic/statistic_result_table/{feature_name}/{feature_name}_result_final_p_value_0.05.csv')

    data = data[data['P-Value'] < 0.01]

    data.to_csv(f'../Statistic/statistic_result_table/{feature_name}/{feature_name}_result_final_p_value_0.01.csv')

    return


def non_feature_selected_SVM(feature_name: str):
    SVM_result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

    ### selected_data_1 contains RBD data, selected_data_0 contains NML data

    selected_data_1 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    i = 0

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    model = svm.SVC(kernel='rbf', C=1, probability=True)

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')

    SVM_result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    SVM_result['Precision'] = np.round(precision_scores.mean(), 2)
    SVM_result['Recall'] = np.round(recall_scores.mean(), 2)
    SVM_result['F1'] = np.round(f1_scores.mean(), 2)

    return pd.DataFrame(SVM_result, index=[1]).to_excel(
        f'/Users/oj/PycharmProjects/Sleep-fMRI/Models/Results/Shen_parcellation/SVM/Non_feature_selected_SVM/{feature_name}_result_rbf.xlsx')


def feature_selected_SVM(feature_name: str, p_value: str):
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
            f'../Statistic/statistic_result_table/Shen_atlas_ancova/{feature_name}/{feature_name}_result_{p_value}.csv')[
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

        model = svm.SVC(kernel='rbf', C=1, probability=True)

        model.fit(np.array(train_data[feature_name].tolist()), train_data['STATUS'])

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

    SVM_result['Accuracy'] = np.round(np.mean(accuracy_score_mean), 2)
    SVM_result['Precision'] = np.round(np.mean(precision_mean), 2)
    SVM_result['Recall'] = np.round(np.mean(recall_mean), 2)
    SVM_result['F1'] = (np.round(np.mean(f1_score_mean), 2))

    pd.DataFrame(SVM_result, index=[1]).to_excel(
        f'./Results/Shen_parcellation/SVM/feature_selected_SVM/{feature_name}/SVM_{feature_name}_result_{p_value}.xlsx')

    return SVM_result


def seed_based_connectivity(matrix, selected_regions):
    # 특정 region 간의 연결성 추출
    connectivity = matrix[0]

    connectivity = connectivity[selected_regions]

    connectivity = connectivity.flatten()

    return connectivity


def vectorize_connectivity(matrix):
    # 특정 region 간의 연결성 추출
    connectivity = matrix[0]

    # 대칭화
    connectivity = (connectivity + connectivity.T) / 2

    # 대각선 0 설정
    np.fill_diagonal(connectivity, 0)

    # 상삼각 행렬 벡터화
    vectorized_fc = connectivity[np.triu_indices(len(connectivity), k=1)]

    return vectorized_fc


non_feature_selected_SVM("fALFF")
