import numpy as np
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import pandas as pd
import nilearn
from nilearn import plotting

Final_data = pd.read_pickle('NML_RBD_data.pkl')

feature_name = 'ALFF'

ttest_data = Final_data[[feature_name, 'STATUS']]

ttest_RBD = ttest_data[ttest_data['STATUS'] == 1]
ttest_HC = ttest_data[ttest_data['STATUS'] == 0]

n_RBD = len(ttest_RBD)
n_NML = len(ttest_HC)

result = []

group_RBD = np.array(ttest_RBD[feature_name].tolist())
group_NML = np.array(ttest_HC[feature_name].tolist())


def local_one_tailed_t_Test(NML_group: list, RBD_group: list):
    for i in range(268):
        NML_feature = NML_group[:, i]
        RBD_feature = RBD_group[:, i]

        ### check whether data follows normal distribution

        p_norm_1 = shapiro(NML_feature).pvalue
        p_norm_0 = shapiro(RBD_feature).pvalue

        # Shapiro's test의 귀무가설은 정규분포를 따른다,이므로 p-value가 0.05이상이면 정규분포를 따른다라는 결론이 도출된다.

        if p_norm_1 > 0.05 and p_norm_0 > 0.05:
            p_levene = levene(NML_feature, RBD_feature).pvalue

            if p_levene > 0.05:
                t_stat, p_value = ttest_ind(NML_feature, RBD_feature, equal_var=True)

                # (t-stat > 0) : NML의 feature값이 RBD feature보다 크다는 뜻. (t-stats < 0): NML의 feature값이 RBD의 feature보다 작다는 뜻.


            else:
                t_stat, p_value = ttest_ind(NML_feature, RBD_feature, equal_var=False)


        else:
            t_stat, p_value_one_tailed = mannwhitneyu(NML_feature, RBD_feature, alternative='two-sided')

        result.append({
            'Feature_Index': i,
            'P-Value': p_value
        })

        return result

    data = pd.read_csv(
        '/Statistic/statistic_result_table/Shen_atlas/REHO/NML_bigger_than_RBD_0.01.csv')
    nodes = pd.read_excel('shen_nodes.xlsx')

    nodes_data = nodes[nodes['label'].isin(data['Feature_Index'])]

    nodes_data.to_csv('NML_bigger_than_RBD_0.01_BrainNetViewer.csv')

    print(nodes_data)

    '''
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
