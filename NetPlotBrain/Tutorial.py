# Import packages
import netplotbrain
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

shen_nodes_coordinates = pd.read_excel(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/BrainNetViewer/shen_nodes_Broamann_label.xlsx')

x_coordinate = shen_nodes_coordinates['X']
y_coordinate = shen_nodes_coordinates['Y']
z_coordinate = shen_nodes_coordinates['Z']

nodes_df = pd.DataFrame()

nodes_df['x'] = x_coordinate.tolist()
nodes_df['y'] = y_coordinate.tolist()
nodes_df['z'] = z_coordinate.tolist()
nodes_df['communities'] = shen_nodes_coordinates['colors'].tolist()

unique_vals = sorted(nodes_df['communities'].unique())
n_colors = len(unique_vals)
cmap = cm.get_cmap('tab10', n_colors)
norm = mcolors.Normalize(vmin=min(unique_vals), vmax=max(unique_vals))
nodes_df['color'] = nodes_df['communities'].apply(lambda x: mcolors.to_hex(cmap(norm(x))))

print(nodes_df['color'])

'''
fig = netplotbrain.plot(
    template='MNI152NLin2009cAsym',
    nodes=nodes_df,
    node_color=nodes_df['communities'].tolist(),
    node_cmap='tab10',
    nodes_type='circles',
)[0]

fig.show()
'''
