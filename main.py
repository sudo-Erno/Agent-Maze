import numpy as np
from maze import Maze
from agent import Agent
import matplotlib.pyplot as plt

# Maze parameters
rows = 5
cols = 5
final = rows - 1, cols - 1
reward_per_step = -0.04
reward_reaching_final = 1.0

# Agent parameters
<<<<<<< HEAD
gamma = 0.95
x0 = 0
y0 = 0
=======
discount_rate = 0.95
>>>>>>> c3b15c998a79870b9e94571340a93fb95c02a378

maze = Maze()

environment = maze.create_maze(final, rows, cols, reward_per_step=reward_per_step, reward_reaching_final=reward_reaching_final)
location_blocks = maze.generate_blocks(2)

<<<<<<< HEAD
agent = Agent(x = x0, y = y0, gamma = gamma, epsilon = 0.7)
=======
agent = Agent(x = 0, y = 0, discount_rate = discount_rate)
>>>>>>> c3b15c998a79870b9e94571340a93fb95c02a378
agent.set_environmet(environment, location_blocks, final)

position_state = False

<<<<<<< HEAD
for i in range(5):
    position_state = agent.move_throught_environment()
=======
game_state, qvalues = agent.move_throught_environment(100)

print(qvalues)
>>>>>>> c3b15c998a79870b9e94571340a93fb95c02a378
