import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import chain
import numpy as np

data = pd.DataFrame(pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features.csv'))

data.loc[:, 'FC'] = data['FC'].apply(ast.literal_eval)

'''
X = data['FC']
y = data['STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

Model = SVC(kernel='linear')
Model.fit(X_train, y_train)
'''
