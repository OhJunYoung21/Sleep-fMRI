import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
import optuna
from xgboost import XGBClassifier

PET_pkl = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_RBD_NML.pkl')


def lower_triangle_vector(mat):
    return mat[np.tril_indices_from(mat, k=-1)]


PET_pkl['FC'] = PET_pkl['FC'].apply(lower_triangle_vector)

different_nodes = pd.DataFrame()
different_nodes['nodes'] = None

feature_difference = []


def non_feature_selected_XGB(feature_name: str):
    result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

    selected_data_1 = PET_pkl.loc[PET_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = PET_pkl.loc[PET_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),

            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**param)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params

    best_params['eval_metric'] = 'logloss'

    model = XGBClassifier(**best_params)

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')

    result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    result['Precision'] = np.round(precision_scores.mean(), 2)
    result['Recall'] = np.round(recall_scores.mean(), 2)
    result['F1'] = np.round(f1_scores.mean(), 2)

    pd.DataFrame(result, index=[1]).to_excel(
        f'./Results/Shen_parcellation/XGB/Non_feature_selected_XGB/XGB_{feature_name}_result_optuna.xlsx')

    return


def feature_selected_XGB(feature_name: str, p_value: str):
    result = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None
    }

    selected_nodes = \
        pd.read_csv(
            f'../Statistic/statistic_result_table/Shen_atlas_ancova/{feature_name}/{feature_name}_result_{p_value}.csv')[
            'Feature_Index']

    ### selected_data_1 contains RBD data, selected_data_0 contains NML data

    selected_data_1 = PET_pkl.loc[PET_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = PET_pkl.loc[PET_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    selected_data_1[feature_name] = selected_data_1[feature_name].apply(lambda x: [x[i] for i in selected_nodes])
    selected_data_0[feature_name] = selected_data_0[feature_name].apply(lambda x: [x[i] for i in selected_nodes])

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**param)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params

    best_params['eval_metric'] = 'logloss'

    model = XGBClassifier(**best_params)

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')

    result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    result['Precision'] = np.round(precision_scores.mean(), 2)
    result['Recall'] = np.round(recall_scores.mean(), 2)
    result['F1'] = np.round(f1_scores.mean(), 2)

    pd.DataFrame(result, index=[1]).to_excel(
        f'./Results/Shen_parcellation/XGB/feature_selected_XGB/{feature_name}/XGB_{feature_name}_result_{p_value}.xlsx')

    return result


non_feature_selected_XGB('fALFF')
