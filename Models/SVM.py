import pandas as pd
import ast
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

Schaefer_features = pd.read_csv(
    '/Users/oj/Desktop/Yoo_Lab/Classification_Features/Schaefer/Schaefer_features_final.csv',
    converters={
        'REHO': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)

'''

---.csv에 문자열로 저장된 데이터를 읽어와주는 코드---

alff_to_process = 'ALFF'  # 처리할 컬럼 이름
Schaefer_features[alff_to_process] = Schaefer_features[alff_to_process].str.split().str.join(',')
reho_to_process = 'REHO'  # 처리할 컬럼 이름
Schaefer_features[reho_to_process] = Schaefer_features[reho_to_process].str.split().str.join(',')
falff_to_process = 'fALFF'  # 처리할 컬럼 이름
Schaefer_features[falff_to_process] = Schaefer_features[falff_to_process].str.split().str.join(',')
Schaefer_features.to_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Schaefer/Schaefer_features_final.csv',
                        index=False)
'''

Schaefer_features['REHO'] = Schaefer_features['REHO'].apply(
    lambda x: np.array(x).flatten())

X = Schaefer_features['REHO']

X_features = {key: [value[4], value[15], value[19], value[20], value[25], value[26], value[32],
                    value[45], value[88], value[100], value[101], value[115]]
              for key, value in X.items()}

y = Schaefer_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
