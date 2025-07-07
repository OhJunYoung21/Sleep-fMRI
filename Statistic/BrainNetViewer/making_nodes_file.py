import pandas as pd
import numpy as np


### Explanation
### 1. make_nodes_file_local method's main purpose is to make nodes file for BrainNetViewer
### 2. It uses data containing nodes which showing significant differences between RBD and NML(Normal)
### 3. output will be stored in same directory

def make_nodes_file_local(feature_name: str):
    significantly_differ_nodes = pd.read_csv(
        f'../statistic_result_table/Shen_atlas_ancova/{feature_name}/RBD_bigger_than_NML/{feature_name}_result_final_0.05.csv')

    template_file = pd.read_excel('./shen_nodes_Broamann_label.xlsx')

    nodes = template_file[template_file['label'].isin(significantly_differ_nodes['Feature_Index'])]

    nodes = nodes.drop(columns=['label'])

    nodes = nodes.rename(columns={'anatomical': 'label'})

    nodes.to_csv(f'./Shen/{feature_name}/RBD_bigger_than_NML/{feature_name}_nodes_0.05.csv', index=False)

    return print(nodes)


def make_edge_file_connectivity():
    significantly_differ_nodes = pd.read_csv(
        f'../statistic_result_table/Shen_atlas_ancova/FC/NML_bigger_than_RBD/FC_result_final_0.0001.csv')

    nodes = (significantly_differ_nodes['Feature_Index'] - 1).tolist()

    '''
    nodes = [x - 1 for x in nodes]
    '''

    matrix = np.zeros((268, 268))

    rows, cols = np.tril_indices(268, -1)

    for k in nodes:
        matrix[rows[k], cols[k]] = 1
        matrix[cols[k], rows[k]] = 1

        print(rows[k] + 1, cols[k] + 1)

    '''

    np.savetxt('Shen/FC/NML_bigger_than_RBD/FC_result_0.05.edge', matrix, fmt='%.0f', delimiter=' ')
    
    '''

    return


make_edge_file_connectivity()
