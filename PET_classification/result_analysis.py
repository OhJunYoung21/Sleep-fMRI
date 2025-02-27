import pandas as pd
import numpy as np
import os
from collections import Counter

t_test_result = pd.read_pickle('statistic_results/mann_whitney_FC.pkl')

result = (t_test_result["Region"][t_test_result['p-value'] < 0.05]).tolist()


def upper_triangular_index(n, vector_index):
    row = int(np.floor((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * vector_index)) / 2))
    col = vector_index + row + 1 - (row * (2 * n - row - 1)) // 2
    return row + 1, col + 1


results = [upper_triangular_index(268, idx) for idx in result]

shen_node_path = "/Users/oj/Desktop/Node_Network_Shen.xlsx"
node_networks = pd.read_excel(shen_node_path)

functional_connectivity = []


def find_network_connection(node1, node2, df):
    """
    주어진 두 개의 노드(node1, node2)에 대해 해당하는 네트워크 정보를 찾아 반환하는 함수.

    Parameters:
        node1 (int): 첫 번째 노드 번호.
        node2 (int): 두 번째 노드 번호.
        df (DataFrame): 노드-네트워크 매핑 데이터프레임.

    Returns:
        tuple: (node1의 네트워크, node2의 네트워크)
    """
    network1 = df.loc[df["Node"] == node1, "Network"].values
    network2 = df.loc[df["Node"] == node2, "Network"].values

    if len(network1) == 0 or len(network2) == 0:
        return "Invalid node number(s)"

    return network1[0], network2[0]


for i, j in results:
    connection = find_network_connection(i, j, node_networks)

    functional_connectivity.append(sorted(connection))


def count_occurrences(lst):
    tuple_list = [tuple(sublist) for sublist in lst]

    # 요소의 등장 횟수 계산
    counts = Counter(tuple_list)

    return counts


count_result = count_occurrences(functional_connectivity)
print(count_result)
