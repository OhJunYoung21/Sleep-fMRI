from nilearn import datasets, plotting, image
import numpy as np
import os
import nibabel as nib
import pandas as pd
from nilearn.datasets import load_mni152_template
from PET_classification.result_analysis import count_occurrences, find_region

### bring shen_atlas file manually

shen_atlas = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen_label_path = '~/Desktop/Node_Network_Shen.xlsx'

shen_label = pd.read_excel(shen_label_path)

t_test_result = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/PET_classification/statistic_results/t_test_fALFF.pkl')

mann_whitney_result = pd.read_pickle(
    '/Users/oj/PycharmProjects/Sleep-fMRI/PET_classification/statistic_results/mann_whitney_fALFF.pkl')

concat_result = pd.concat([t_test_result, mann_whitney_result], axis=1)

result = (concat_result["Region"][concat_result['p-value'] < 0.05]).tolist()

node_networks = shen_label

nodes = count_occurrences(result)

shen_img = image.load_img(shen_atlas)

shen_nodes = np.array(nodes)

shen_nodes_list = list(shen_nodes)

selected_node = image.math_img("img * np.isin(img, {})".format(shen_nodes_list),
                               img=shen_img)

plotting.plot_roi(selected_node, bg_img=load_mni152_template(),
                  title="Shen Atlas - Network 4 Nodes", display_mode="ortho", cut_coords=(0, 0, 0), cmap="autumn")
plotting.show()
