import numpy as np

arr = np.random.rand(3, 10)
#print(arr)
#print(arr.shape)

alpha = 0.0001
arr = arr.T * alpha

print(arr)
