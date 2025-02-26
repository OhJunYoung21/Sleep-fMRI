import pandas as pd
import numpy as np

t_test_result = pd.read_pickle('t_test_result.pkl')

result = (t_test_result["Region"][t_test_result['p-value'] < 0.05]).tolist()


def upper_triangular_index(n, vector_index):
    row = int(np.floor((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * vector_index)) / 2))
    col = vector_index + row + 1 - (row * (2 * n - row - 1)) // 2
    return row + 1, col + 1


results = [upper_triangular_index(268, idx) for idx in result]

print(results)
