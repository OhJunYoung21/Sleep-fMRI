from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import ast
import numpy as np

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

Shen_features['REHO'] = Shen_features['REHO'].apply(
    lambda x: np.array(x).flatten())

X = np.array(Shen_features['REHO'])
y = Shen_features['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
