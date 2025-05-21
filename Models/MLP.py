import pandas as pd
import ast
import os
from nilearn import plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

NML_RBD_pkl = pd.read_pickle('../Statistic/NML_RBD_data.pkl')


def feature_selected_MLP(feature_name: str, p_value: str):
    accuracy_score_mean = []
    f1_score_mean = []
    precision_mean = []
    recall_mean = []

    result = {
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

        model = MLPClassifier(
            hidden_layer_sizes=(100,),  # 은닉층 구조 (100개의 뉴런을 가진 1개의 은닉층)
            activation='relu',  # 활성화 함수 (ReLU)
            solver='lbfgs',  # 옵티마이저 (Adam 최적화 알고리즘)
            learning_rate='constant',  # 학습률 스케줄
            learning_rate_init=0.01,  # 초기 학습률
            max_iter=1000,  # 최대 반복 횟수
            random_state=42
            # 난수 시드
        )

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

    result['Accuracy'] = np.round(np.mean(accuracy_score_mean), 2)
    result['Precision'] = np.round(np.mean(precision_mean), 2)
    result['Recall'] = np.round(np.mean(recall_mean), 2)
    result['F1'] = (np.round(np.mean(f1_score_mean), 2))

    pd.DataFrame(result, index=[1]).to_excel(
        f'./Results/MLP/Statistic_feature_selected_MLP/MLP_{feature_name}_result_{p_value}.xlsx')

    return result


feature_selected_MLP("ALFF", "0.05")
feature_selected_MLP("ALFF", "0.01")


def non_feature_selected_MLP(feature_name: str):
    accuracy_score_mean = []
    f1_score_mean = []
    precision_mean = []
    recall_mean = []

    result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

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

        model = MLPClassifier(
            hidden_layer_sizes=(100,),  # 은닉층 구조 (100개의 뉴런을 가진 1개의 은닉층)
            activation='relu',  # 활성화 함수 (ReLU)
            solver='lbfgs',  # 옵티마이저 (Adam 최적화 알고리즘)
            learning_rate='constant',  # 학습률 스케줄
            learning_rate_init=0.01,  # 초기 학습률
            max_iter=500,  # 최대 반복 횟수
            random_state=42
            # 난수 시드
        )

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

    result['Accuracy'] = np.round(np.mean(accuracy_score_mean), 2)
    result['Precision'] = np.round(np.mean(precision_mean), 2)
    result['Recall'] = np.round(np.mean(recall_mean), 2)
    result['F1'] = (np.round(np.mean(f1_score_mean), 2))

    pd.DataFrame(result, index=[1]).to_excel(
        f'./Results/MLP/Non_statistic_feature_selected_MLP/MLP_{feature_name}_result.xlsx')

    return result
