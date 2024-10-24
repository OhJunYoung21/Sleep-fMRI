import pandas as pd
import ast
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

upgraded_features = pd.DataFrame()

upgraded_features['up_ALFF'] = None
upgraded_features['STATUS'] = None
upgraded_features['ALFF'] = None

juelich_features = pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/BASC/BASC_features_final.csv',
                               converters={
                                   'STATUS': ast.literal_eval,
                                   'FC': ast.literal_eval}
                               )

alff_to_process = 'ALFF'  # 처리할 컬럼 이름
juelich_features[alff_to_process] = juelich_features[alff_to_process].str.split().str.join(',')
reho_to_process = 'REHO'  # 처리할 컬럼 이름
juelich_features[reho_to_process] = juelich_features[reho_to_process].str.split().str.join(',')
falff_to_process = 'fALFF'  # 처리할 컬럼 이름
juelich_features[falff_to_process] = juelich_features[falff_to_process].str.split().str.join(',')
juelich_features.to_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/BASC/BASC_features_final.csv',
                        index=False)
'''
juelich_features['FC'] = juelich_features['FC'].apply(
    lambda x: np.array(x).flatten())

X = np.array(juelich_features['FC'])
y = juelich_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

result_matrix = confusion_matrix(y_test, y_pred)

print(acc)
'''
