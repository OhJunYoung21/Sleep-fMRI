import pandas as pd
import numpy as np
import os

data = pd.read_csv('statistic_result_table/Shen_atlas/NML_bigger_RBD_FC/NML_bigger_RBD_FC_final_0.05.csv')

nodes_list = data['Unnamed: 0'].tolist()


def make_edges_for_visualization(node_list: list, n: int):
    matrix = np.zeros((n, n))

    tril_rows, tril_cols = np.tril_indices(n, k=-1)

    for k in node_list:
        row, col = tril_rows[k], tril_cols[k]
        matrix[row][col] = 1
        matrix[col][row] = 1

    return matrix


matrix = make_edges_for_visualization(nodes_list, 268)

np.savetxt('/Users/oj/Desktop/BrainNetViewer/Data/ExampleFiles/Shen268/Edges/NML_bigger_RBD_FC_final_0.05.edge', matrix,
           delimiter=' ', fmt='%.0f')


def make_nodes_for_connectivity(node_list: list):
    shen_node = pd.read_excel('shen_nodes.xlsx')

    return
