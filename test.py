from random import choice
import random
import numpy as np

env = np.zeros((8, 4))
env_map = np.zeros_like(env, dtype=np.str)

env[0][0] = 1
env[1][0] = 1
env[2][0] = 1

actual_state = [0, 0]
next_states = [
    [actual_state[0] - 1, actual_state[1]],
    [actual_state[0], actual_state[1] + 1],
    [actual_state[0] + 1, actual_state[1]],
    [actual_state[0], actual_state[1] - 1]
]

values = list()

for state in next_states:
    row, col = state
    values.append(env[row][col])

max_state_value_index = values.index(max(values))

if max_state_value_index == 2:
    env_map[actual_state[0]][actual_state[1]] = "Down"

print(env.shape[0])