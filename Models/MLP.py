import optuna
import pandas as pd
import ast
import os
from nilearn import plotting
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedKFold
import numpy as np
import shap
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

NML_RBD_pkl = pd.read_pickle('../Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_NML_RBD.pkl')


def feature_selected_MLP(feature_name: str, p_value: str):
    result = {
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

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    def objective(trial):
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(64,), (128,), (64, 64)])
        alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=500,
            random_state=42
        )

        score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params

    mlp = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        max_iter=500,
        random_state=42
    )

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    accuracy_scores = cross_val_score(mlp, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(mlp, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(mlp, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(mlp, X, y, cv=cv, scoring='recall')

    result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    result['Precision'] = np.round(precision_scores.mean(), 2)
    result['Recall'] = np.round(recall_scores.mean(), 2)
    result['F1'] = np.round(f1_scores.mean(), 2)

    return result


def non_feature_selected_MLP(feature_name: str):
    result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

    selected_data_1 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (32, 16)],
        'alpha': [0.01, 0.1, 1.0],
        'learning_rate_init': [0.0001, 0.001]
    }

    # MLP 모델 정의
    mlp = MLPClassifier(max_iter=500, random_state=42)

    # 내부 튜닝을 포함한 모델 정의
    clf = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5)

    accuracy_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    precision_scores = cross_val_score(clf, X, y, cv=5, scoring='precision')
    recall_scores = cross_val_score(clf, X, y, cv=5, scoring='recall')

    result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    result['Precision'] = np.round(precision_scores.mean(), 2)
    result['Recall'] = np.round(recall_scores.mean(), 2)
    result['F1'] = np.round(f1_scores.mean(), 2)

    pd.DataFrame(result, index=[1]).to_excel(
        f'./Results/Shen_parcellation/MLP/feature_selected_MLP/{feature_name}/MLP_{feature_name}_result.xlsx')

    return
