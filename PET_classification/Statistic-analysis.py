import numpy as np
from scipy.stats import ttest_ind, shapiro, levene
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import brunnermunzel

pet_data = pd.read_pickle('PET_shen_static.pkl')

feature_name = 'REHO'

ttest_data = pet_data[[feature_name, 'STATUS']]

ttest_positive = ttest_data[ttest_data['STATUS'] == 1]
ttest_negative = ttest_data[ttest_data['STATUS'] == 0]

ttest_result = []

group_positive = np.array(ttest_positive[feature_name].tolist())
group_negative = np.array(ttest_negative[feature_name].tolist())

for i in range(268):
    positive_feature = group_positive[:, i]
    negative_feature = group_negative[:, i]

    ### check whether data follows normal distribution

    p_norm_1 = shapiro(positive_feature).pvalue
    p_norm_0 = shapiro(negative_feature).pvalue

    if p_norm_1 > 0.05 and p_norm_0 > 0.05:  # 두 그룹 모두 정규성을 만족하는 경우
        # 등분산성 검정 (Levene’s test)
        p_levene = levene(positive_feature, negative_feature).pvalue
        equal_var = p_levene > 0.05  # 등분산 여부 확인

        # t-test 수행
        t_stat, p_value = ttest_ind(positive_feature, negative_feature, equal_var=equal_var)
        test_used = "t-test (equal_var={})".format(equal_var)

    else:  # 하나라도 정규성을 따르지 않으면 비모수 검정 수행
        u_stat, p_value = mannwhitneyu(positive_feature, negative_feature, alternative='two-sided')
        test_used = "Mann-Whitney U test"

    # 결과 저장
    ttest_result.append({"Region": i, "Test": test_used, "p-value": p_value})

t_test_result = pd.DataFrame(ttest_result)

print(t_test_result[t_test_result['Test'] == "Mann-Whitney U test"])
