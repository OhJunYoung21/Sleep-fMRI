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

alff_data = pd.read_excel('alff_diff_data.xlsx')

alff_regions_normality = alff_data['significant_diff']
alff_regions_mann = alff_data['significant_diff_mann']

alff_nan_ttest = next((i for i, x in enumerate(alff_regions_normality) if np.isnan(x)), len(alff_regions_normality))
alff_regions_normality = alff_regions_normality[:alff_nan_ttest].tolist()

alff_nan_mann = next((i for i, x in enumerate(alff_regions_mann) if np.isnan(x)), len(alff_regions_mann))
alff_regions_mann = alff_regions_mann[:alff_nan_mann].tolist()

alff_regions = alff_regions_mann + alff_regions_normality

alff_regions = [int(i) for i in alff_regions]

X_features = np.array({key: [value[i] for i in alff_regions if i < len(value)] for key, value in X.items()})

y = Schaefer_features['STATUS']

svm_model = svm.SVC(kernel='linear')

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(svm_model, X.tolist(), y, cv=k_fold)

# 결과 출력
print("Cross-validation scores:", np.round(scores, 2))
print("Mean accuracy:", np.mean(scores))
print("Standard deviation of accuracy:", np.std(scores))
