import numpy as np
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import pandas as pd
import nilearn
from nilearn import plotting
import statsmodels.formula.api as smf

NML_data = pd.read_pickle('../Static_Feature_Extraction/Schaefer_features/schaefer_covariate_NML.pkl')
RBD_data = pd.read_pickle('../Static_Feature_Extraction/Schaefer_features/schaefer_covariate_RBD.pkl')

Final_data = pd.concat([NML_data, RBD_data], axis=0)


def ANCOVA_test(feature_name: str):
    result = []

    for i in range(300):
        nodes_feature = [k[i] for k in Final_data[feature_name]]

        ancova_data = pd.DataFrame({'group': Final_data['STATUS'].tolist(),
                                    'sex': Final_data['sex'].tolist(),
                                    'age': Final_data['age'].tolist(),
                                    f'{feature_name}': np.array(nodes_feature)
                                    })
        model = smf.ols(f'{feature_name} ~ group +  age + sex', data=ancova_data).fit()

        if model.pvalues.group < 0.05:
            result.append({
                f'node_{i}': i,
                'p_value':
                    model.pvalues})
        else:
            continue

    return result


print(ANCOVA_test('FC'))

'''
data = pd.read_csv(
    '/Statistic/statistic_result_table/Shen_atlas/REHO/NML_bigger_than_RBD_0.01.csv')
nodes = pd.read_excel('shen_nodes.xlsx')

nodes_data = nodes[nodes['label'].isin(data['Feature_Index'])]

nodes_data.to_csv('NML_bigger_than_RBD_0.01_BrainNetViewer.csv')

print(nodes_data)


for j in range(len(result_FC)):
    row, col = result_FC.iloc[j]['connected_nodes']
    matrix[row - 1][col - 1] = result_FC.iloc[j]['T-Value']
    matrix[col - 1][row - 1] = result_FC.iloc[j]['T-Value']

plotting.plot_matrix(
    matrix,
    labels=None,
    figure=(15, 13),
    vmax=1,
    vmin=-1,
    title="Connectivity bigger in NML than RBD(p-value 0.001)",
)
plotting.show()
'''
