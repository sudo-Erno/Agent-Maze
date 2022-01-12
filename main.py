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
gamma = 0.95
x0 = 0
y0 = 0

maze = Maze()

environment = maze.create_maze(final, rows, cols, reward_per_step=reward_per_step, reward_reaching_final=reward_reaching_final)
location_blocks = maze.generate_blocks(2)

agent = Agent(x = x0, y = y0, gamma = gamma, epsilon = 0.7)
agent.set_environmet(environment, location_blocks, final)

position_state = False

for i in range(5):
    position_state = agent.move_throught_environment()