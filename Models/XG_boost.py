import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
import optuna
from xgboost import XGBClassifier

NML_RBD_pkl = pd.read_pickle('../Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_NML_RBD.pkl')

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

    selected_data_1 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 1, [feature_name, 'STATUS']]
    selected_data_0 = NML_RBD_pkl.loc[NML_RBD_pkl['STATUS'] == 0, [feature_name, 'STATUS']]

    i = 0

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
        f'./Results/Shen_parcellation/XGB/Non_feature_selected_XGB/XGB_{feature_name}_result_optuna.xlsx')

    return


def feature_selected_XGB(feature_name: str, p_value: str):
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

        model = XGBClassifier()

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
        f'./Results/Shen_parcellation/XGB/feature_selected_XGB/{feature_name}/XGB_{feature_name}_result_{p_value}.xlsx')

    return result


non_feature_selected_XGB("FC")
