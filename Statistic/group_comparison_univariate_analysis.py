import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import shuffle
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import bct

Final_data = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_PET.pkl')


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


def ANCOVA_test(feature_name: str, compare: str):
    X = np.array(Final_data[feature_name].tolist())  # shape: [n_subjects, 35778]

    n_features = X.shape[1]

    t_obs = np.zeros(n_features)
    p_values = np.zeros(n_features)

    group_vec = Final_data['STATUS'].values
    sex_vec = Final_data['sex'].values
    age_vec = Final_data['age'].values

    for i in range(n_features):
        feature = X[:, i]
        df = pd.DataFrame({
            'group': group_vec,
            'sex': sex_vec,
            'age': age_vec,
            'edge': feature
        })

        model = smf.ols('edge ~ group + sex + age', data=df).fit()
        t_obs[i] = model.tvalues['group']
        p_values[i] = model.pvalues['group']

    '''
    if compare == 'hyper':
        result = np.where((p_values < 0.05) & (t_obs > 0))[0].tolist()
    else:
        result = np.where((p_values < 0.05) & (t_obs < 0))[0].tolist()
    '''

    result = np.where(p_values < 0.05)[0].tolist()

    return result


def ANCOVA_test_FDR_bh(feature_name: str):
    X = np.array(Final_data[feature_name].tolist())  # shape: [n_subjects, 35778]

    n_features = 268
    t_obs = np.zeros(n_features)
    p_values = []
    group_vec = Final_data['STATUS'].values
    sex_vec = Final_data['sex'].values
    age_vec = Final_data['age'].values

    for i in range(n_features):
        feature = X[:, i]
        df = pd.DataFrame({
            'group': group_vec,
            'sex': sex_vec,
            'age': age_vec,
            'feature': feature
        })

        model = smf.ols('feature ~ group + sex + age', data=df).fit()
        t_obs[i] = model.tvalues['group']
        p_values.append(model.pvalues['group'])

    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    return np.where(pvals_corrected < 0.05)


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

                    ### accumulate degrees obtained from each threshold, respectively.

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

    print("processing...")

    return tfnbs_matrix


def get_significant_edges(tfnbs_matrix, t_null_distribution, alpha=0.05):
    n = tfnbs_matrix.shape[0]
    tri_idx = np.tril_indices(n, k=-1)
    obs_vals = tfnbs_matrix[tri_idx]

    null_vals = np.array([null_mat[tri_idx] for null_mat in t_null_distribution])  # shape: [n_perm, n_edges]
    p_vals = np.mean(null_vals >= obs_vals, axis=0)  # p-value 계산

    significant_edges = []
    for idx, p in enumerate(p_vals):
        if p < alpha:
            i, j = tri_idx[0][idx], tri_idx[1][idx]
            significant_edges.append((i, j))

    return significant_edges


def TFNBS_permutation(feature_name: str):
    X = np.array(Final_data[feature_name].tolist())  # shape: [n_subjects, 35778]
    seeds = 5000

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
        delayed(perm_single_TFNBS_run)(seed, X, group_vec, sex_vec, age_vec) for seed in range(seeds)
    )

    return get_significant_edges(tfnbs_matrix, t_null_distribution, alpha=0.05)


def compute_nCPL(fc_matrix, threshold: float, n_random=42):
    """
        Parameters:
        - fc_matrix: 2D numpy array (e.g., 268x268)
        - sparsity: proportion of strongest edges to keep (e.g., 0.2)
        - n_random: number of random graphs to generate for null model

        Returns:
        - nCPL: normalized characteristic path length
        """
    fc_matrix = np.abs(fc_matrix)
    n = fc_matrix.shape[0]

    # 1. Thresholding → Binarize
    triu_idx = np.triu_indices(n, k=1)
    edge_weights = fc_matrix[triu_idx]
    n_edges_to_keep = int(threshold * len(edge_weights))
    threshold_value = np.sort(edge_weights)[-(n_edges_to_keep)]

    A = (fc_matrix >= threshold_value).astype(int)
    np.fill_diagonal(A, 0)

    # 2. Real graph path length
    D_real = bct.distance_bin(A)
    L_real, _, _, _, _ = bct.charpath(D_real, include_infinite=False)

    # 3. Random graphs
    L_rand_list = []
    for _ in range(n_random):
        A_rand, _ = bct.randmio_und(A.copy(), 10)  # 10 swaps per edge
        D_rand = bct.distance_bin(A_rand)
        L_rand, _, _, _, _ = bct.charpath(D_rand)
        L_rand_list.append(L_rand)

    L_rand_mean = np.mean(L_rand_list)

    # 4. Compute nCPL
    nCPL = L_real / L_rand_mean

    return nCPL


def make_file_BNV(feature_name: str, compare: str):
    regions = ANCOVA_test(feature_name, compare)

    regions = [i + 1 for i in regions]

    labels_data = pd.read_excel(
        '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/BrainNetViewer/shen_nodes_Broamann_label.xlsx')

    labels_data = labels_data[labels_data['label'].isin(regions)].drop(columns=['label'])

    return labels_data.to_excel(
        f'/Users/oj/Desktop/BrainNetViewer/Data/ExampleFiles/Shen268/Nodes/ANCOVA_analysis/{feature_name}_{compare}.xlsx')
