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

schaefer_pkl = pd.read_pickle('schaefer_data_pkl')

for i in range(20):
    status_1_data = schaefer_pkl[schaefer_pkl['STATUS'] == 1]
    status_0_data = schaefer_pkl[schaefer_pkl['STATUS'] == 0]
    # Select only the REHO and STATUS columns
    selected_data_1 = status_1_data[['REHO', 'STATUS']]
    selected_data_0 = status_0_data[['REHO', 'STATUS']]
    # Split 80% of the data for training
    train_data_1 = selected_data_1.sample(frac=0.8, random_state=42 + i)
    train_data_0 = selected_data_0.sample(frac=0.8, random_state=42 + i)

    test_data_1 = selected_data_1.drop(train_data_1.index)
    test_data_0 = selected_data_0.drop(train_data_0.index)

    train_data = pd.concat([train_data_1, train_data_0])
    test_data = pd.concat([test_data_1, test_data_0])

    X_train = np.array(train_data['REHO'].tolist())
    y_train = train_data['STATUS']
    X_test = np.array(test_data['REHO'].tolist())
    y_test = test_data['STATUS']

    model = svm.SVC(kernel='rbf', C=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"{i + 1}th score : {accuracy_score(y_test, y_pred):.2f}")
