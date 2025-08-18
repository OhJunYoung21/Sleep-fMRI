import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RepeatedKFold

NML_RBD_pkl = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Static_Feature_Extraction/Shen_features/shen_RBD_HC_18_parameters.pkl')


def lower_triangle_vector(mat):
    return mat[np.tril_indices_from(mat, k=-1)]


NML_RBD_pkl['FC'] = NML_RBD_pkl['FC'].apply(lower_triangle_vector)


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

    i = 0

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    def rf_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

        clf = RandomForestClassifier(**param, random_state=42)
        score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(rf_objective, n_trials=30)

    best_params = study.best_trial.params

    model = RandomForestClassifier(**best_params)

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')

    SVM_result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    SVM_result['Precision'] = np.round(precision_scores.mean(), 2)
    SVM_result['Recall'] = np.round(recall_scores.mean(), 2)
    SVM_result['F1'] = np.round(f1_scores.mean(), 2)

    pd.DataFrame(SVM_result, index=[1]).to_excel(
        f'./Results/Shen_parcellation/RF/Non_feature_selected_RF/RF_{feature_name}_result.xlsx')

    return


def feature_selected_RF(feature_name: str, p_value: str):
    RF_result = {
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

    i = 0

    full_data = pd.concat([selected_data_0, selected_data_1], axis=0).reset_index(drop=True)
    X = np.array(full_data[feature_name].tolist())
    y = full_data['STATUS'].values

    def rf_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

        clf = RandomForestClassifier(**param, random_state=42)
        score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(rf_objective, n_trials=30)

    best_params = study.best_trial.params

    model = RandomForestClassifier(**best_params)

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')

    RF_result['Accuracy'] = np.round(accuracy_scores.mean(), 2)
    RF_result['Precision'] = np.round(precision_scores.mean(), 2)
    RF_result['Recall'] = np.round(recall_scores.mean(), 2)
    RF_result['F1'] = np.round(f1_scores.mean(), 2)

    pd.DataFrame(RF_result, index=[1]).to_excel(
        f'./Results/Shen_parcellation/RF/feature_selected_RF/{feature_name}/RF_{feature_name}_result_{p_value}.xlsx')

    return RF_result


feature_selected_RF("FC", '0.05')
