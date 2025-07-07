import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import shuffle
import statsmodels.formula.api as smf

Final_data = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_NML_RBD.pkl')


def originate_FC(low_tril: list):
    n = 268

    reconstructed = np.zeros((n, n))

    # 대각선 제외한 하삼각 인덱스
    i, j = np.tril_indices(n, k=-1)

    # 값 채우기
    reconstructed[i, j] = low_tril[0]

    # 대칭 복원 (상삼각 부분에 복사)
    reconstructed[j, i] = reconstructed[i, j]

    return reconstructed


def apply_threshold(matrix: np.ndarray):
    abs_matrix = np.abs(matrix)

    flattened = abs_matrix.flatten()

    sorted_values = np.sort(flattened)[::-1]

    threshold = sorted_values[133]

    return threshold


Final_data = Final_data.reset_index(drop=True)

Final_data['Original_FC'] = Final_data['FC'].apply(originate_FC)
Final_data['binarized_FC'] = [(np.abs(i) >= apply_threshold(i)).astype(int) for i in Final_data['Original_FC']]


def ANCOVA_test(feature_name: str):
    X = np.array(Final_data[feature_name].tolist())  # shape: [n_subjects, 35778]
    n_perm = 5000

    if feature_name == "FC":
        n_edges = 35778
    else:
        n_edges = 268

    t_obs = np.zeros(n_edges)
    group_vec = Final_data['STATUS'].values
    sex_vec = Final_data['sex'].values
    age_vec = Final_data['age'].values

    for i in range(n_edges):
        edge_feature = X[:, i]
        df = pd.DataFrame({
            'group': group_vec,
            'sex': sex_vec,
            'age': age_vec,
            'edge': edge_feature
        })

        model = smf.ols('edge ~ group + sex + age', data=df).fit()
        t_obs[i] = model.tvalues['group']

    def perm_single_run(seed, X, group_vec, sex_vec, age_vec):
        shuffled = shuffle(group_vec, random_state=seed)
        t_vals = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            edge_feature = X[:, i]
            df = pd.DataFrame({
                'group': shuffled,
                'sex': sex_vec,
                'age': age_vec,
                'edge': edge_feature
            })
            model = smf.ols('edge ~ group + sex + age', data=df).fit()
            t_vals[i] = model.tvalues['group']

        return t_vals

    seeds = np.arange(n_perm)

    # 3. permutation t-value 저장
    t_null_dist = Parallel(n_jobs=-1)(
        delayed(perm_single_run)(seed, X, group_vec, sex_vec, age_vec) for seed in seeds
    )

    # 결과를 NumPy 배열로 변환 (n_perm x n_edges)
    t_null_dist = np.array(t_null_dist)

    p_values = np.mean(np.abs(t_null_dist) >= np.abs(t_obs[np.newaxis, :]), axis=0)

    # 5. 유의한 edge 표시
    significant_edges = p_values < 0.05

    # 6. 결과 저장
    result = pd.DataFrame({
        'Feature_Index': np.arange(1, n_edges + 1),
        't_value': t_obs,
        'p_value_perm': p_values,
        'significant_perm': significant_edges
    })

    print(result['Feature_Index'][result['significant_perm'] == True])

    return


ANCOVA_test("fALFF")
