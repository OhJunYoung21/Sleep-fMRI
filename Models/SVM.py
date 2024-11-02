import pandas as pd
import ast
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

AAL_features = pd.read_csv(
    '/Users/oj/Desktop/Yoo_Lab/Classification_Features/AAL/AAL_features_final.csv',
    converters={
        'fALFF': ast.literal_eval,
        'REHO': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)

X = np.array(AAL_features['REHO'])
y = AAL_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
