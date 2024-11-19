import pandas as pd
import ast
import os
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
from Visualization.T_test import check_normality, student_t_test, welch_t_test, mann_whitney_test, check_variance

schaefer_pkl = pd.read_pickle('../schaefer_data_pkl')


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


feature_difference = []
for i in range(20):
    status_1_data = schaefer_pkl[schaefer_pkl['STATUS'] == 1]
    status_0_data = schaefer_pkl[schaefer_pkl['STATUS'] == 0]
    # Select only the REHO and STATUS columns
    selected_data_1 = status_1_data[['REHO', 'STATUS']]
    selected_data_0 = status_0_data[['REHO', 'STATUS']]
    # Split 80% of the data for training
    train_data_1 = selected_data_1.sample(frac=0.7, random_state=42 + i)
    train_data_0 = selected_data_0.sample(frac=0.7, random_state=42 + i)

    test_data_1 = selected_data_1.drop(train_data_1.index)
    test_data_0 = selected_data_0.drop(train_data_0.index)

    train_data = pd.concat([train_data_1, train_data_0])
    test_data = pd.concat([test_data_1, test_data_0])

    rbd_data = train_data['REHO'][train_data['STATUS'] == 1]
    hc_data = train_data['REHO'][train_data['STATUS'] == 0]

    result_list = statistic(rbd_data, hc_data)

    feature_difference.append(result_list)

    result = avoid_duplication(feature_difference)

    train_data['REHO'] = train_data['REHO'].apply(lambda x: [x[i] for i in result])
    test_data['REHO'] = test_data['REHO'].apply(lambda x: [x[i] for i in result])

    model = svm.SVC(kernel='linear', C=1)
    model.fit(np.array(train_data['REHO'].tolist()), train_data['STATUS'])

    accuracy = model.score(np.array(test_data['REHO'].tolist()), test_data['STATUS'])

    print(f"{i + 1}th accuracy : {accuracy:.2f}")