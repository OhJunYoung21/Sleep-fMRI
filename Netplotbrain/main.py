import netplotbrain
import pandas as pd
import numpy as np
import templateflow.api as tf
import itertools
import matplotlib.pyplot as plt

template = 'MNI152NLin2009cAsym'
nodes_df = pd.DataFrame(data={'x': [40, 10, 30, -15, -25],
                              'y': [50, 40, -10, -20, 20],
                              'z': [20, 30, -10, -15, 30],
                              'communities': [1, 1, 1, 2, 2],
                              'degree_centrality': [1, 1, 0.2, 0.8, 0.4]})

netplotbrain.plot(nodes=nodes_df, arrowaxis=None)
plt.show()
