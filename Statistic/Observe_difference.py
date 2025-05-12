import pandas as pd
import numpy as np
import os

df_NML = pd.read_excel('/Users/oj/Desktop/Yoo_Lab/Data_covariate.xlsx', sheet_name='NML')

df_NML = df_NML.drop(columns=['Unnamed: 0'])

df_RBD = pd.read_excel('/Users/oj/Desktop/Yoo_Lab/Data_covariate.xlsx', sheet_name='RBD')

df_RBD = df_RBD.drop(columns=['Unnamed: 0'])

print(df_RBD)
