from random import choice
import random
import numpy as np

env = np.zeros((5, 3))

for i in range(env.shape[0] - 1):
    print(f"{i = }")