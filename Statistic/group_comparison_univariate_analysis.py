import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import shuffle
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

Final_data = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Static_Feature_Extraction/Shen_features/shen_RBD_HC_18_parameters.pkl')


def lower_triangle_vector(mat):
    return mat[np.tril_indices_from(mat, k=-1)]


Final_data['FC_vectorized'] = Final_data['FC'].apply(lower_triangle_vector)


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


'''
Final_data = Final_data.reset_index(drop=True)

Final_data['Original_FC'] = Final_data['FC'].apply(originate_FC)
'''


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

    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    print(np.where(rejected == True))

    # 5. 유의한 edge 표시
    significant_edges = p_values < 0.05

    # 6. 결과 저장
    result = pd.DataFrame({
        'Feature_Index': np.arange(1, n_edges + 1),
        't_value': t_obs,
        'p_value_perm': p_values,
        'significant_perm': significant_edges
    })

    return


def ANCOVA_test_FDR_bh(feature_name: str):
    X = np.array(Final_data[feature_name].tolist())  # shape: [n_subjects, 35778]

    n_features = 35778

    t_obs = np.zeros(n_features)
    p_values = []
    group_vec = Final_data['STATUS'].values
    sex_vec = Final_data['sex'].values
    age_vec = Final_data['age'].values

    for i in range(n_features):
        node_feature = X[:, i]
        df = pd.DataFrame({
            'group': group_vec,
            'sex': sex_vec,
            'age': age_vec,
            'node': node_feature
        })

        model = smf.ols('node ~ group + sex + age', data=df).fit()
        t_obs[i] = model.tvalues['group']
        p_values.append(model.pvalues['group'])

    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    result = list(np.where(rejected & (t_obs < 0)))

    return result


def compute_tfnbs_from_tmatrix(t_matrix, E=0.5, H=1.0):
    tfnbs_matrix = np.zeros_like(t_matrix)

    t_max = np.max(t_matrix)

    thresholds = np.arange(0, t_max, np.round(t_max / 100, 2))

    for threshold in thresholds:
        adj = (t_matrix > threshold).astype(int)
        G = nx.from_numpy_array(adj)

        for cluster in nx.connected_components(G):
            nodes = list(cluster)
            cluster_score = 0
            for node in nodes:
                deg = G.degree(node)
                intensity = sum((t_matrix[node, neighbor] - threshold)
                                for neighbor in G.neighbors(node))
                cluster_score += (deg ** E) * (intensity ** H)
            for i in nodes:
                for j in nodes:
                    if i < j and adj[i, j]:
                        tfnbs_matrix[i, j] += cluster_score
                        tfnbs_matrix[j, i] += cluster_score
    return tfnbs_matrix


def perm_single_TFNBS_run(seed, X, group_vec, sex_vec, age_vec):
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

    mat = np.eye(268)

    tri_idx = np.tril_indices(268, k=-1)

    mat[tri_idx] = t_vals

    mat = mat + mat.T - np.diag(np.diag(mat))

    tfnbs_matrix = compute_tfnbs_from_tmatrix(mat, E=0.5, H=1.0)

    return tfnbs_matrix


def TFNBS_permutation(feature_name: str):
    X = np.array(Final_data[feature_name].tolist())  # shape: [n_subjects, 35778]
    seeds = 100

    n_edges = 35778

    t_obs = np.zeros(35778)

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

    mat = np.eye(268)

    tri_idx = np.tril_indices(268, k=-1)

    mat[tri_idx] = t_obs

    mat = mat + mat.T - np.diag(np.diag(mat))

    tfnbs_matrix = compute_tfnbs_from_tmatrix(mat, E=0.5, H=1.0)

    t_null_distribution = Parallel(n_jobs=-1)(
        delayed(perm_single_TFNBS_run)(seed, X, group_vec, sex_vec, age_vec) for seed in seeds
    )

    return


edges = ANCOVA_test_FDR_bh('FC_vectorized')[0]

print(edges)

'''
template_data = pd.read_excel(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/BrainNetViewer/shen_nodes_Broamann_label.xlsx')

template_data = template_data.iloc[nodes, :]

template_data.drop(columns=['label'], inplace=True)

template_data.to_csv(
    '/Users/oj/Desktop/BrainNetViewer/Data/ExampleFiles/Shen268/Nodes/ANCOVA_analysis/RBD_hypo_connectivity/ALFF_nodes.node',
    sep='\t',
    index=False, header=False)
'''
