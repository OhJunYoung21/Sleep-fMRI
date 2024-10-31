import pandas as pd
import ast
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

Shen_features = pd.read_csv(
    '/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features_final.csv',
    converters={
        'ALFF': ast.literal_eval,
        'fALFF': ast.literal_eval,
        'REHO': ast.literal_eval,
        'FC': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)

Shen_features['FC'] = Shen_features['FC'].apply(
    lambda x: np.array(x).flatten())

X = np.array(Shen_features['FC'])
y = Shen_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

clf = svm.SVC()

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_

y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(grid_search.best_params_)
print(accuracy)
