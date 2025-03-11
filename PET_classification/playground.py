import pandas as pd
import os
import numpy as np
from PET_classification.result_analysis import find_network_connection, node_networks
from collections import Counter

feature_name = "tril_FC"

t_test_result = pd.read_pickle(
    f'/Users/oj/PycharmProjects/Sleep-fMRI/PET_classification/statistic_results/t_test_{feature_name}.pkl')

'''
mann_whitney_result = pd.read_pickle(
    f'/Users/oj/PycharmProjects/Sleep-fMRI/PET_classification/statistic_results/mann_whitney_{feature_name}.pkl')

concat_result = pd.concat([t_test_result, mann_whitney_result], axis=0)

result = (concat_result["Region"][concat_result['p-value'] < 0.05]).tolist()
'''


def find_index(n, vector_index):
    row = int(np.floor((np.sqrt(8 * vector_index + 1) - 1) / 2))
    col = vector_index - (row * (row + 1)) // 2
    return row + 1, col + 1


connections = [find_index(268, i) for i in t_test_result["Region"].tolist()]

connections_regions = [find_network_connection(i, j, node_networks) for i, j in connections]

new_counter = Counter()

number = 0

for key, value in Counter(connections_regions).items():
    # 키를 (a, b)와 (b, a)로 동일하게 처리
    a, b = key

    if (b, a) in new_counter:
        new_counter[(b, a)] += value  # (b, a)와 기존에 있던 값을 합침
    else:
        new_counter[(a, b)] += value  # (a, b)를 새로 추가
