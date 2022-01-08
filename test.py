from random import choice
import random
import numpy as np

my_zeros_arr = np.zeros((4, 4))

my_arr = np.array(my_zeros_arr, dtype=np.str)

for i in range(my_arr.shape[0]):
    for j in range(my_arr.shape[1]):
        my_arr[i][j] = "x"
        if i == 1 and j == 1:
            my_arr[i][j] = "A"

print(my_arr)