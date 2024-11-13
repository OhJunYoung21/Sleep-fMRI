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
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score

Schaefer_features = pd.read_csv(
    '/Users/oj/Desktop/Yoo_Lab/Classification_Features/Schaefer/Schaefer_features_final.csv',
    converters={
        'REHO': ast.literal_eval,
        'ALFF': ast.literal_eval,
        'FC': ast.literal_eval,
        'fALFF': ast.literal_eval,
        'STATUS': ast.literal_eval
    }
)

'''
fc_to_process = 'FC'  # 처리할 컬럼 이름
Schaefer_features[fc_to_process] = Schaefer_features[fc_to_process].str.strip(',')
Schaefer_features[fc_to_process] = Schaefer_features[fc_to_process].str.split().str.join(',')
alff_to_process = 'ALFF'  # 처리할 컬럼 이름
Schaefer_features[alff_to_process] = Schaefer_features[alff_to_process].str.split().str.join(',')
reho_to_process = 'REHO'  # 처리할 컬럼 이름
Schaefer_features[reho_to_process] = Schaefer_features[reho_to_process].str.split().str.join(',')
falff_to_process = 'fALFF'  # 처리할 컬럼 이름
Schaefer_features[falff_to_process] = Schaefer_features[falff_to_process].str.split().str.join(',')
Schaefer_features.to_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Schaefer/Schaefer_features_final.csv',
                         index=False)
'''

Schaefer_features['FC'] = Schaefer_features['FC'].apply(
    lambda x: np.array(x).flatten())

X = Schaefer_features['FC']

y = Schaefer_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = np.array(X_train)
X_test = np.array(X_test)


'''
reho_data = pd.read_excel('reho_Schaefer_data.xlsx')

reho_welch = reho_data['welch'].tolist()
reho_student = reho_data['student'].tolist()
reho_mann = reho_data['mann_whitneyu'].tolist()

reho_welch_index = next((i for i, x in enumerate(reho_welch) if np.isnan(x)), len(reho_welch))
reho_welch = reho_welch[:reho_welch_index]

reho_student_index = next((i for i, x in enumerate(reho_student) if np.isnan(x)), len(reho_student))
reho_student = reho_student[:reho_student_index]

reho_mann_index = next((i for i, x in enumerate(reho_mann) if np.isnan(x)), len(reho_mann))
reho_mann = reho_mann[:reho_mann_index]

reho_regions = reho_student + reho_mann + reho_welch

reho_regions = [int(i) for i in reho_regions]
'''

## X와 y의 샘플수를 맞추기 위해서 다운샘플링을 진행하는 코드이다.
'''
for i in range(100):
    rbd_X = X[y == 1]
    rbd_y = y[y == 1]

    hc_X = X[y == 0]
    hc_y = y[y == 0]

    np.random.seed(42 + i)

    rbd_sample_indices = np.random.choice(rbd_X.index, size=len(hc_X), replace=False)

    rbd_X_sample = rbd_X.loc[rbd_sample_indices]
    rbd_y_sample = rbd_y.loc[rbd_sample_indices]

    X_balanced = pd.concat([rbd_X_sample, hc_X])
    y_balanced = pd.concat([rbd_y_sample, hc_y])

    X_balanced = X_balanced.reset_index(drop=True)
    y_balanced = y_balanced.reset_index(drop=True)

    ### X_features = np.array([row[reho_regions] for row in X_balanced.values])

    svm_model = svm.SVC(kernel='rbf')

    # KFold 설정 (5-폴드 교차검증)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # 교차검증 수행
    scores = cross_val_score(svm_model, X_balanced, y_balanced, cv=kfold, scoring='accuracy')

    print(f"{i + 1}th F1-score: {np.mean(scores):.2f}")
'''
