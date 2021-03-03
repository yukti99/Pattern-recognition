import numpy as np
print("NUMPY version = ",np.__version__)


arr = np.array([1,2,3,4,5])  # can pass list, tuple or any array-like object to this function
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr3)
print(type(arr3))
print(arr3.ndim)