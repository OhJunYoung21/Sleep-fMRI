import numpy as np
from scipy.stats import ttest_ind, shapiro, levene
import pandas as pd

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

    ### PET_positive와 PET_negative사이에서 실제로 차이가 날 확률이 5%이상인 feature들에 대해서만 다음 작업을 수행한다.

    if p_norm_1 > 0.05 and p_norm_0 > 0.05:
        p_levene = levene(positive_feature, negative_feature).pvalue

        if p_levene > 0.05:
            t_stat, p_value = ttest_ind(positive_feature, negative_feature, equal_var=True)
        else:
            t_stat, p_value = np.nan, np.nan
    else:
        t_stat, p_value, p_levene = np.nan, np.nan, np.nan

    ttest_result.append({
        'Feature_Index': i,
        'Shapiro_Positive_p': p_norm_1,
        'Shapiro_Negative_p': p_norm_0,
        'Levene_p': p_levene,
        'T-Statistic': t_stat,
        'P-Value': p_value
    })

t_test_result = pd.DataFrame(ttest_result)

filtered_df = t_test_result[t_test_result['P-Value'].fillna(1) <= 0.05]  # NaN을 1로 대체 후 비교


print(filtered_df)
