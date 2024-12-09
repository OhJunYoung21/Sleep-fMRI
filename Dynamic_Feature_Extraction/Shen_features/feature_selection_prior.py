import pandas as pd
import os

shen_data = pd.read_pickle('shen_dynamic.pkl')

shen_data["prior_ALFF"] = shen_data["ALFF"].apply(lambda x:
                                                  [x[0][i - 1] for i in
                                                   [41, 43, 59, 66, 67, 69, 71, 73, 74, 175, 177, 200, 201, 204, 206,
                                                    209, 210, 240]])


