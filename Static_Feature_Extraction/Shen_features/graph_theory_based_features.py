import scipy.io
import pickle
import pandas as pd
import bct
import numpy as np

data = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/statistic_result_table/Shen_atlas_ancova/Data/shen_PET.pkl')


class GraphMetric:
    def __init__(self, connectivity):
        self.connectivity = connectivity
        self.n = connectivity.shape[0]

    def calculate_degree(self):
        density = 0.2
        A_thr = bct.threshold_proportional(self.connectivity, density)

        deg = bct.degrees_und(A_thr)
        return deg

    def calculate_path_length(self):
        density = 0.2
        A_thr = bct.threshold_proportional(self.connectivity, density)

        length_matrix = bct.weight_conversion(A_thr, 'lengths')

        D, B = bct.distance_wei(length_matrix)

        lambda_, efficiency, ecc, radius, diameter = bct.charpath(D)

        result = [lambda_, efficiency]

        return result

    def calculate_BC(self):
        density = 0.2

        A_thr = bct.threshold_proportional(self.connectivity, density)

        L = bct.weight_conversion(A_thr, 'lengths')  # e.g., L = 1 / A_thr (내부적으로 처리)

        BC_wei = bct.betweenness_wei(L)

        return BC_wei

    def calculate_CC(self):
        density = 0.2

        A_thr = bct.threshold_proportional(self.connectivity, density)

        C_wu = bct.clustering_coef_wu(A_thr)
        return C_wu

    def calculate_LCC(self):
        density = 0.2

        W_thr = bct.threshold_proportional(self.connectivity, 0.20)

        E_loc_nodes_w = bct.efficiency_wei(W_thr, local=True)

        return E_loc_nodes_w
