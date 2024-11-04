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
        'REHO': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)

X = AAL_features['REHO']

X_features = {key: [value[0][4], value[0][5], value[0][8], value[0][9], value[0][24], value[0][25], value[0][26],
                    value[0][33], value[0][34], value[0][35], value[0][42], value[0][43], value[0][56], value[0][87]]
              for key, value in X.items()}
y = AAL_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
