import numpy as np
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import pandas as pd

Final_data = pd.read_pickle('NML_RBD_shen_data.pkl')

feature_name = 'FC'

ttest_data = Final_data[[feature_name, 'STATUS']]

ttest_RBD = ttest_data[ttest_data['STATUS'] == 1]
ttest_HC = ttest_data[ttest_data['STATUS'] == 0]

ttest_result = []

feature_list = []

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

                if t_stat > 0:
                    p_value_one_tailed_greater = p_value / 2
                else:
                    # 양의 방향에서는 t_stat가 음수일 경우, p-value는 1에서 반으로 나누어 계산
                    p_value_one_tailed_smaller = 1 - p_value / 2
            else:
                t_stat, p_value = ttest_ind(NML_feature, RBD_feature, equal_var=False)

                if t_stat > 0:
                    p_value_one_tailed_greater = p_value / 2
                else:
                    # 양의 방향에서는 t_stat가 음수일 경우, p-value는 1에서 반으로 나누어 계산
                    p_value_one_tailed_smaller = 1 - p_value / 2
        else:
            t_stat, p_value_one_tailed = mannwhitneyu(NML_feature, RBD_feature, alternative='two-sided')

            if t_stat > 0:
                p_value_one_tailed_greater = p_value / 2
            else:
                # 양의 방향에서는 t_stat가 음수일 경우, p-value는 1에서 반으로 나누어 계산
                p_value_one_tailed_smaller = 1 - p_value / 2

        ttest_result['greater'].append({
            'Feature_Index': i,
            'P-Value': p_value_one_tailed_greater
        })

        ttest_result['less'].append({
            'Feature_Index': i,
            'P-Value': p_value_one_tailed_smaller
        })

    return ttest_result


def local_two_tailed_t_Test(NML_group: list, RBD_group: list):
    for i in range(35778):
        NML_feature = NML_group[:, i][0]
        RBD_feature = RBD_group[:, i][0]

        ### check whether data follows normal distribution

        p_norm_1 = shapiro(NML_feature).pvalue
        p_norm_0 = shapiro(RBD_feature).pvalue

        # Shapiro's test의 귀무가설은 정규분포를 따른다,이므로 p-value가 0.05이상이면 정규분포를 따른다라는 결론이 도출된다.

        if p_norm_1 > 0.05 and p_norm_0 > 0.05:
            p_levene = levene(NML_feature, RBD_feature).pvalue

            if p_levene > 0.05:
                t_stat, p_value = ttest_ind(NML_feature, RBD_feature, equal_var=True)
            else:
                t_stat, p_value = ttest_ind(NML_feature, RBD_feature, equal_var=False)
        else:
            t_stat, p_value = mannwhitneyu(NML_feature, RBD_feature, alternative='two-sided')

        ttest_result.append({
            'Feature_Index': i,
            'P-Value': p_value
        })

    return ttest_result


ttest_result = local_two_tailed_t_Test(group_NML, group_RBD)

significant_features = [d['Feature_Index'] for d in ttest_result if d['P-Value'] < 0.000005]

print(len(significant_features))

'''
shen_nodes = pd.read_excel('/Users/oj/Desktop/shen_nodes_test.xlsx', sheet_name='Basic_nodes')

shen_nodes['colors'] = shen_nodes['label'].isin(significant_features).astype(int)

shen_nodes.loc[shen_nodes['colors'] == 1, 'colors'] = shen_nodes['Network']

shen_nodes.to_excel('/Users/oj/Desktop/shen_nodes_test_1.xlsx')
'''
