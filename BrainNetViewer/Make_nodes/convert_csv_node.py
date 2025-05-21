import pandas as pd
import os

feature_name = 'fALFF'

data = pd.read_csv(
    f'/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/statistic_result_table/{feature_name}/RBD_bigger_than_NML_0.05_BrainNetViewer.csv')

label_info = pd.read_excel('shen_nodes_Broamann_label.xlsx')

# label을 기준으로 anatomical 이름을 매핑
data['label_name'] = data['label'].map(label_info.set_index('label')['anatomical'])

temple = data.drop('Unnamed: 0', axis=1)

temple.to_excel(f'./{feature_name}/RBD_bigger_than_NML_0.05_name_drop.xlsx')
