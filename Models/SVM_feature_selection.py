import pandas as pd
import ast
import os
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from Visualization.T_test import check_normality, student_t_test, welch_t_test, mann_whitney_test, check_variance
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV

shen_pkl = pd.read_pickle('../Dynamic_Feature_Extraction/Shen_features/shen_268_dynamic.pkl')

feature_nodes = [1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 17, 19, 21, 22, 30, 31, 41,
                 43, 47, 48, 49, 50, 55, 56, 67, 69, 70, 71, 73, 74, 85, 86, 90, 96,
                 111, 112, 115, 116, 134, 138, 139, 141, 142, 143, 147, 154, 157,
                 164, 175, 177, 182, 184, 193, 196, 199, 200, 201, 203, 204, 206,
                 209, 210, 222, 223, 225, 227, 239, 240, 242, 246, 247]


def prior_extraction_connectivity(matrix, selected_regions):
    # 특정 region 간의 연결성 추출
    connectivity = matrix[0][np.ix_(selected_regions, selected_regions)]

    # 대칭화
    connectivity = (connectivity + connectivity.T) / 2

    # 대각선 0 설정
    np.fill_diagonal(connectivity, 0)

    # 상삼각 행렬 벡터화
    vectorized_fc = connectivity[np.triu_indices(len(selected_regions), k=1)]

    return vectorized_fc


def vectorize_connectivity(matrix, selected_regions):
    # 특정 region 간의 연결성 추출
    connectivity = matrix[0]

    # 대칭화
    connectivity = (connectivity + connectivity.T) / 2

    # 대각선 0 설정
    np.fill_diagonal(connectivity, 0)

    # 상삼각 행렬 벡터화
    vectorized_fc = connectivity[np.triu_indices(len(selected_regions), k=1)]

    return vectorized_fc


shen_pkl['prior_FC'] = shen_pkl['FC'].apply(lambda x: prior_extraction_connectivity(x, feature_nodes))
shen_pkl['normal_FC'] = shen_pkl['FC'].apply(lambda x: vectorize_connectivity(x, feature_nodes))
shen_pkl['prior_REHO'] = shen_pkl['REHO'].apply(lambda x: [x[0][i - 1] for i in feature_nodes])
shen_pkl['prior_ALFF'] = shen_pkl['ALFF'].apply(lambda x: [x[0][i - 1] for i in feature_nodes])
shen_pkl['prior_fALFF'] = shen_pkl['fALFF'].apply(lambda x: [x[0][i - 1] for i in feature_nodes])


def avoid_duplication(nested_list):
    unique_elements = set()
    for sublist in nested_list:
        unique_elements.update(sublist)

    # 집합을 다시 리스트로 변환
    result = list(unique_elements)

    return result


def statistic(rbd_data, hc_data):
    mann_whitneyu, welch, student = check_variance(np.array(rbd_data.tolist()),
                                                   np.array(hc_data.tolist()))

    student_test = student_t_test(student, np.array(rbd_data.tolist()),
                                  np.array(hc_data.tolist()))
    welch_test = welch_t_test(welch, np.array(rbd_data.tolist()),
                              np.array(hc_data.tolist()))
    mann_test = mann_whitney_test(mann_whitneyu, np.array(rbd_data.tolist()),
                                  np.array(hc_data.tolist()))

    return np.unique(mann_test + student_test + welch_test).tolist()


accuracy_score_mean = []
f1_score_mean = []
precision_mean = []
recall_mean = []
feature_difference = []

feature_name = "prior_FC"

selected_data_1 = shen_pkl.loc[shen_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
selected_data_0 = shen_pkl.loc[shen_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

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

    '''
    train_data[feature_name] = [item[0] for item in train_data[feature_name]]
    test_data[feature_name] = [item[0] for item in test_data[feature_name]]
    '''

    '''
    train_data[feature_name] = train_data[feature_name].apply(lambda x: [x[i] for i in result])
    test_data[feature_name] = test_data[feature_name].apply(lambda x: [x[i] for i in result])
    '''

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

print(f"average of accuracy : {np.round(np.mean(accuracy_score_mean), 2)}")
print(f"average of f1-score : {np.round(np.mean(f1_score_mean), 2)}")
print(f"average of recall : {np.round(np.mean(recall_mean), 2)}")
print(f"average of precision : {np.round(np.mean(precision_mean), 2)}")
