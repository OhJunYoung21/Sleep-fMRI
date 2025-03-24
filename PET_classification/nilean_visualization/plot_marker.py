from nilearn import datasets, plotting, image
import numpy as np
import os
import nibabel as nib
import pandas as pd
from nilearn.datasets import load_mni152_template
from PET_classification.result_analysis import count_occurrences, find_region

### bring shen_atlas file manually

shen_atlas = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen_label_path = '/Users/oj/Desktop/Yoo_Lab/atlas/Shen268_10network.xlsx'

shen_label = pd.read_excel(shen_label_path)

shen_nodes_label = shen_label["node"]

shen_atlas_img = image.load_img(shen_atlas)

## bring statistic data

feature_name = "Reho"

t_test_result = pd.read_pickle(
    f'/Users/oj/PycharmProjects/Sleep-fMRI/PET_classification/statistic_results/t_test_{feature_name}.pkl')

mann_whitney_result = pd.read_pickle(
    f'/Users/oj/PycharmProjects/Sleep-fMRI/PET_classification/statistic_results/mann_whitney_{feature_name}.pkl')

concat_result = pd.concat([t_test_result, mann_whitney_result], axis=0)

result = (concat_result["Region"][concat_result['p-value'] < 0.05]).tolist()

# calculate coords from shen_atlas
# shen_nodes_labels match shen_atlas.nii's get_fdata

coord = plotting.find_parcellation_cut_coords(shen_atlas_img, background_label=0)


def find_network(node, df):
    network = int(df.loc[df["node"] == (node + 1), "network"].values)

    return network


networks = [find_network(j, shen_label) for j in result]

plotting.plot_markers(
    title = "Reho nodes showing difference between RBD and NML",
    node_values=[1 for j in range(len(networks))],
    node_coords=[coord[k] for k in result],
    node_size='auto'

)
plotting.show()
