import numpy as np
from scipy.stats import ttest_ind, shapiro, levene
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import brunnermunzel

pet_data = pd.read_pickle('preprocessed_shen_data.pkl')

feature_name = 'tril_FC'

ttest_data = pet_data[[feature_name, 'STATUS']]

ttest_positive = ttest_data[ttest_data['STATUS'] == 1]
ttest_negative = ttest_data[ttest_data['STATUS'] == 0]

group_positive = np.array(ttest_positive[feature_name].tolist())
group_negative = np.array(ttest_negative[feature_name].tolist())

statistic_result = []

for i in range(35778):
    ### i번째 열에 해당하는 모든 행의 값을 positive,negative_feature에 저장한다.

    positive_feature = group_positive[:, i]
    negative_feature = group_negative[:, i]

    ### check whether data follows normal distribution

    p_norm_1 = shapiro(positive_feature).pvalue
    p_norm_0 = shapiro(negative_feature).pvalue

    p_levene = levene(positive_feature, negative_feature).pvalue
    equal_var = p_levene > 0.05  # 등분산 여부 확인

    u_stat, p_value = mannwhitneyu(positive_feature, negative_feature, alternative='two-sided')
    test_used = "Mann-Whitney U test"

    '''
    # t-test 수행
    t_stat, p_value = ttest_ind(positive_feature, negative_feature, equal_var=equal_var)
    test_used = "t-test (equal_var={})".format(equal_var)
    '''
    statistic_result.append({"Region": i, "Test": test_used, "p-value": p_value})

statistic_result = pd.DataFrame(statistic_result)

statistic_result.to_pickle(f'./statistic_results/mann_whitney_{feature_name}.pkl')
