# Import packages
import netplotbrain
import pandas as pd

shen_nodes_coordinates = pd.read_excel(
    '/Users/oj/PycharmProjects/Sleep-fMRI/Statistic/BrainNetViewer/shen_nodes_Broamann_label.xlsx')

x_coordinate = shen_nodes_coordinates['X']
y_coordinate = shen_nodes_coordinates['Y']
z_coordinate = shen_nodes_coordinates['Z']

nodes_df = pd.DataFrame()

nodes_df['x'] = x_coordinate.tolist()
nodes_df['y'] = y_coordinate.tolist()
nodes_df['z'] = z_coordinate.tolist()
nodes_df['communities'] = shen_nodes_coordinates['colors']

fig = netplotbrain.plot(template='MNI152NLin2009cAsym',
                        template_style='surface',
                        template_voxelsize=3,
                        nodes=nodes_df,
                        nodes_color=nodes_df['communities'],
                        nodes_scale=10,
                        arrowaxis=None)[0]
fig.show()
