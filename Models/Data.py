import pandas as pd
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import chain
import numpy as np
from Static_Feature_Extraction import Shen_features

data = pd.read_pickle('./Static_Feature_Extraction/Shen_features/shen_268_CNN.pkl')