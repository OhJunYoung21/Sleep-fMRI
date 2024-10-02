import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import chain
import numpy as np

data = pd.DataFrame(pd.read_csv('/Users/oj/Desktop/Yoo_Lab/Classification_Features/Shen/Shen_features.csv'))


