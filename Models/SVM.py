import pandas as pd
import ast
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

upgraded_features = pd.DataFrame()

upgraded_features['up_ALFF'] = None
upgraded_features['STATUS'] = None
upgraded_features['ALFF'] = None

Schaefer_features = pd.read_csv(
    '/Users/oj/Desktop/Yoo_Lab/Classification_Features/Schaefer/Schaefer_features_final.csv',
    converters={
        'FC': ast.literal_eval,
        'ALFF': ast.literal_eval,
        'fALFF': ast.literal_eval,
        'REHO': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)
'''
X = np.array(Schaefer_features['REHO'])
y = Schaefer_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())


'''
Schaefer_features[['REHO']] = Schaefer_features[['REHO']].apply(
    lambda x: np.array(x).flatten())

X = np.array(Schaefer_features['REHO'])
y = Schaefer_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

clf = svm.SVC(kernel='linear', C=0.1, gamma=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

result_matrix = confusion_matrix(y_test, y_pred)

print(acc)
