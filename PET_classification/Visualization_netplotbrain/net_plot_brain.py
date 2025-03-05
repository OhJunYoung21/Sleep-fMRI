import netplotbrain
from netplotbrain import plot
import nibabel as nib
import pandas as pd
import os
import numpy as np
from scipy.ndimage import measurements
from itertools import chain
from PET_classification.result_analysis import count_edges

shen_atlas = '/Users/oj/Desktop/Yoo_Lab/atlas/shen_2mm_268_parcellation.nii'

shen_img = nib.load(shen_atlas)
shen_data = shen_img.get_fdata()

region_ids = np.unique(shen_data)[1:]
region_coords = []

for region_id in region_ids:
    region_mask = shen_data == region_id
    center_of_mass = measurements.center_of_mass(region_mask)
    region_coords.append(center_of_mass)

nodes_df = pd.DataFrame({
    'id': region_ids,
    'x': [coord[0] for coord in region_coords],
    'y': [coord[1] for coord in region_coords],
    'z': [coord[2] for coord in region_coords],
    'label': [f'Region {int(id)}' for id in region_ids]
})

nodes_df['Index'] = nodes_df.index + 1

indices = list(chain.from_iterable(count_edges))

filtered_df = nodes_df[nodes_df['Index'].isin(indices)]

print(filtered_df)


edges_df = pd.DataFrame(count_edges, columns=['source', 'target'])

netplotbrain.plot(template='MNI152NLin2009cAsym', nodes=filtered_df, edges=edges_df)
