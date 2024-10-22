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

shen_features = pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features_final.csv',
                            converters={'ALFF': ast.literal_eval,
                                        'fALFF': ast.literal_eval,
                                        'REHO': ast.literal_eval,
                                        'STATUS': ast.literal_eval,
                                        'FC': ast.literal_eval}
                            )
shen_features['ALFF'] = shen_features['ALFF'].apply(
    lambda x: np.array(x).flatten())

X = np.array(shen_features['ALFF'])
y = shen_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_pre_train, y_pre_train)
importance = model.feature_importances_

values_array = np.array(importance.tolist())

threshold = np.percentile(values_array, 90)

top_10_percent_indices = np.where(values_array >= threshold)[0]

upgraded_features['ALFF'] = X_pre_train.tolist()
upgraded_features['up_ALFF'] = upgraded_features['ALFF'].apply(lambda x: [x[i] for i in top_10_percent_indices])

print(upgraded_features.head())

'''
print(upgraded_features)


clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

result_matrix = confusion_matrix(y_test, y_pred)

print(result_matrix,len(y_pred))


target = shen_features['FC'][1]

plotting.plot_matrix(target,
                     figure=(9, 7),
                     vmax=0.5,
                     vmin=-0.5,
                     title="Functional Connectivity")
plotting.show()
'''
