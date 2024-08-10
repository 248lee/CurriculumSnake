import torch as th
import numpy as np
a = th.from_numpy(np.array([[[2, 3, 4], [5, 6, 7]], [[8, 9, 10], [11, 12, 13]]]))
row_idx = th.from_numpy(np.array([0, 1]))
col_idx = th.from_numpy(np.array([0, 1, 2]))
print(a[row_idx, col_idx])
print(len(a[row_idx, col_idx]))