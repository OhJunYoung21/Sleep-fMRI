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

# coord에는 268의 region의 중심좌표(3차원으로 표현)을 담고 있는 numpy 배열이다.

coord = plotting.find_parcellation_cut_coords(shen_atlas_img, background_label=0)

sorted_coord = np.array([coord[np.where(shen_nodes_label == i)[0][0]] for i in shen_nodes_label])

plotting.plot_markers(
    node_values=[1, 1, 1],
    node_coords=[sorted_coord[0], sorted_coord[1], sorted_coord[2]],
    node_size='auto'
)
plotting.show()
