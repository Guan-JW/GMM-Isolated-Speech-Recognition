import numpy as np

a = np.array([[1,2,3],[1,2,1]])
b = np.array([2,1])[:, np.newaxis]
print(a/b)
