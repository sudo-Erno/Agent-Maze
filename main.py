import numpy as np
from maze import Maze
from agent import Agent
import matplotlib.pyplot as plt

# Maze parameters
rows = 5
cols = 5
final = rows - 1, cols - 1

# Agent parameters
reward_per_step = -0.04
reward_reaching_final = 1.0
gamma = 0.95

maze = Maze()

environment = maze.create_maze(final, rows, cols)
environment = maze.generate_blocks(2)

agent = Agent(reward_per_step, reward_reaching_final, gamma = gamma)
agent.set_environmet(environment, final)

agent.move_throught_environment("right")

# print(agent.actual_coords_x, agent.actual_coords_y)