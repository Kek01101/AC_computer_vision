import numpy as np

# Demonstrating loading of numpy zips for now
arr = np.load('cameraCalNums.npz')

obj = arr['arr_0']
img = arr['arr_1']
