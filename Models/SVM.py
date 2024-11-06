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

Schaefer_features['ALFF'] = Schaefer_features['ALFF'].apply(
    lambda x: np.array(x).flatten())

X = Schaefer_features['ALFF']

y = Schaefer_features['STATUS']

alff_data = pd.read_excel('alff_diff_data.xlsx')

alff_regions_normality = alff_data['significant_diff']
alff_regions_mann = alff_data['significant_diff_mann']

alff_nan_ttest = next((i for i, x in enumerate(alff_regions_normality) if np.isnan(x)), len(alff_regions_normality))
alff_regions_normality = alff_regions_normality[:alff_nan_ttest].tolist()

alff_nan_mann = next((i for i, x in enumerate(alff_regions_mann) if np.isnan(x)), len(alff_regions_mann))
alff_regions_mann = alff_regions_mann[:alff_nan_mann].tolist()

alff_regions = alff_regions_mann + alff_regions_normality

alff_regions = [int(i) for i in alff_regions]

print(len(alff_regions))

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

    X_features = np.array([row[alff_regions] for row in X_balanced.values])

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3,
                                                        random_state=42)

    model = svm.SVC(kernel='poly', C=1.0, gamma=0.1)

    model.fit(np.array(X_train.tolist()), np.array(y_train.tolist()))

    y_pred = model.predict(np.array(X_test.tolist()))

    print(f"{i + 1}th Accuracy: {accuracy_score(y_test, y_pred):.2f}")
