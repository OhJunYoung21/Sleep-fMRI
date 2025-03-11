import pandas as pd
import numpy as np
import os
from collections import Counter

shen_node_path = "/Users/oj/Desktop/Yoo_Lab/atlas/Shen268_10network.xlsx"
node_networks = pd.read_excel(shen_node_path)


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
    network1 = df.loc[df["node"] == node1, "network"].values
    network2 = df.loc[df["node"] == node2, "network"].values

    if len(network1) == 0 or len(network2) == 0:
        return "Invalid node number(s)"

    return network1[0], network2[0]


def find_region(node, df):
    region = df.loc[df["Node"] == (node + 1), "Network"].values

    return region


'''
for i, j in results:
    connection = find_network_connection(i, j, node_networks)

    functional_connectivity.append(sorted(connection))
'''


def count_occurrences(lst):
    networks = []

    for j in lst:
        network = find_region(j, node_networks)
        networks.append(network.item())

    return networks
